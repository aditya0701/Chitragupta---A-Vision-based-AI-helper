"""Trigger engine — the zero-LLM-cost per-tick 'thinking'.

Runs pure arithmetic on the world doc and returns a list of trigger events.
The reasoning model is only woken when this returns something (or a frame
arrived / the user spoke). This is the same design principle the old
timers.py followed — wall-clock math is free, tokens are not — extended to
the whole system.

Trigger kinds:
  expectation_due   a time-anchored expectation passed its deadline while
                    still open ("rice not confirmed started, deadline gone")
  stale_task        an in_progress task hasn't been mentioned by any caption,
                    fact, or tool call in STALENESS_S — earns a check-in
                    question ("still on the chicken?")
  proposal_pending  a plan was proposed and the user never said yes or no.
                    Blocks everything downstream, so it re-asks rather than
                    waiting to be noticed.
  wanted_found      something on the find list has been seen and the user has
                    not been told. Coalesced: one event for all of them.
  wanted_stuck      a search has run long enough with no result, or too many
                    UNCLEAR answers, to be worth asking the user about.

Event-anchored expectations never appear here: their firing condition can
only be judged by the model against the current frame, so they ride along in
the rendered doc and the tick prompt instead.
"""

from __future__ import annotations

import time

from . import config, worlddoc


def check(doc: dict) -> list[dict]:
    now = time.time()
    events: list[dict] = []

    for exp in worlddoc.open_expectations(doc):
        if exp["anchor"] == "time" and exp["due_ts"] and exp["due_ts"] <= now:
            # Claim it before any awaited model call happens downstream —
            # same double-fire lesson as timers.mark_firing().
            exp["status"] = "fired"
            exp["fired_ts"] = now
            events.append({
                "kind": "expectation_due",
                "priority": exp["priority"],
                "expectation": exp,
                "text": (
                    f"Expectation '{exp['description']}' (set {worlddoc.fmt_ts(exp['created_ts'])}, "
                    f"due {worlddoc.fmt_ts(exp['due_ts'])}) has passed its deadline without being "
                    f"confirmed done."
                ),
            })

    # A plan proposed and never answered. The assistant is blocked on the user
    # and the user does not know it — the one silence that costs them something
    # rather than sparing them. Re-armed each time, so it keeps asking rather
    # than giving up on a plan that may just have been missed over a noisy hob.
    proposal = doc.get("proposal")
    if proposal:
        raised = proposal.get("raised_ts") or proposal.get("ts") or now
        if now - raised >= config.PROPOSAL_RERAISE_S:
            proposal["raised_ts"] = now  # claim it, same as a fired expectation
            events.append({
                "kind": "proposal_pending",
                "priority": "normal",
                "proposal": proposal,
                "text": (
                    f"The plan '{proposal['title']}' ({len(proposal['steps'])} steps) was "
                    f"proposed at {worlddoc.fmt_ts(proposal['ts'])} and the user has not "
                    f"said yes or no. Nothing is being tracked until they do. Ask them "
                    f"once, briefly — do not read the whole plan out again."
                ),
            })

    # Something the user asked us to find has been seen, and they have not been
    # told. Note what this does NOT read: the caption. Detection already
    # happened in worlddoc.fold_wanted during the tick's first write window, so
    # this is pure doc arithmetic like everything else here — which means it
    # also fires from poll(), where there is no frame at all. A find that
    # arrives just as the camera goes dark still reaches the user.
    #
    # ONE event for ALL unannounced finds, never one per item. Three things
    # spotted in a single pantry shot are one sentence — "onions are in the
    # wire basket, tomatoes in the bowl behind them" — not three utterances
    # fired back to back.
    pending = worlddoc.unannounced_finds(doc)
    if pending:
        events.append({
            "kind": "wanted_found",
            # Deliberately NOT "high". High means physical risk and buys a
            # politeness-gate bypass this does not need: phase 3e already
            # treats any event as important. Inflating it here would be the
            # one-flag-many-consequences mistake that broke v1's camera.
            "priority": "normal",
            "items": pending,
            "text": "; ".join(
                f"'{w['item']}' — asked for at {worlddoc.fmt_ts(w['asked_ts'])}, "
                f"seen at {worlddoc.fmt_ts(w['found_ts'])}: {w['where']}"
                for w in pending
            ),
        })

    # A search that is going nowhere. Silence here is the failure mode: the
    # assistant looks, fails, looks again, and the user hears nothing at all
    # until they think to ask. Both causes get the same consequence — ask the
    # user — with wording that matches which one it is.
    for w in worlddoc.open_wanted(doc):
        if w.get("stuck_raised"):
            continue
        misses = int(w.get("misses") or 0)
        unclears = int(w.get("unclears") or 0)
        if unclears >= config.WANTED_UNCLEAR_ASKS:
            reason = (
                f"'{w['item']}' has come back UNCLEAR {unclears} times — the camera "
                f"keeps half-seeing something it cannot confirm. Ask them to hold "
                f"steady on it, or move closer, for a second."
            )
        elif misses >= config.WANTED_STUCK_ASKS:
            reason = (
                f"'{w['item']}' has not been visible in {misses} frames since "
                f"{worlddoc.fmt_ts(w['asked_ts'])}. Ask them where they usually keep "
                f"it — you have been looking in the wrong place."
            )
        else:
            continue
        w["stuck_raised"] = True  # claim it: ask once, not every tick
        events.append({
            "kind": "wanted_stuck",
            "priority": "low",
            "wanted": w,
            "text": reason,
        })

    for task in doc["tasks"]:
        if task["status"] != "in_progress":
            continue
        last = task.get("last_mention_ts") or doc.get("session_started") or now
        # Don't nag about a task that's just quietly waiting on its own
        # open time-anchored expectation — the deadline will speak for it.
        covered = any(
            e.get("task_id") == task["id"] and e["anchor"] == "time"
            for e in worlddoc.open_expectations(doc)
        )
        if not covered and now - last >= config.STALENESS_S:
            task["last_mention_ts"] = now  # reset so it doesn't re-fire every tick
            events.append({
                "kind": "stale_task",
                "priority": "low",
                "task": task,
                "text": (
                    f"No update on in-progress task '{task['content']}' for "
                    f"{int((now - last) // 60)} minutes — consider asking the user for a status."
                ),
            })

    return events


def may_speak_unprompted(doc: dict, priority: str = "normal") -> bool:
    """Politeness budget for speech the user didn't ask for. High priority
    always passes; everything else waits out the minimum gap since the last
    utterance."""
    if priority == "high":
        return True
    return time.time() - (doc.get("last_spoken_ts") or 0.0) >= config.MIN_UNPROMPTED_GAP_S


def mark_spoke(doc: dict):
    doc["last_spoken_ts"] = time.time()


def mark_user_turn(doc: dict, text: str = ""):
    """Record that the user just asked for something. Tracked separately from
    last_spoken_ts because the two pull in opposite directions: speaking should
    make the assistant quieter, being asked should make it more forthcoming.

    The TEXT is stored, not just the timestamp. Stage 2's speech prompt used to
    receive only "[User last spoke] 143s ago" while being asked to judge
    whether a caption answered what they wanted — a test with no operand. See
    worlddoc._empty_doc's last_user_msg for the session where that cost 65
    seconds of knowing and not saying.
    """
    doc["last_user_turn_ts"] = time.time()
    if (text or "").strip():
        doc["last_user_msg"] = text.strip()[:400]


def in_followup_window(doc: dict) -> bool:
    """True while a tick is still allowed to volunteer something on the back of
    a recent user request, regardless of the politeness gap.

    Deliberately NOT folded into may_speak_unprompted(): poll()'s stale-task
    nags also route through that function, and a nag firing seconds after a
    conversation is the exact thing the gap is right to suppress. Only the tick
    path — where the model is reacting to what it can actually see — opts in.
    """
    return time.time() - (doc.get("last_user_turn_ts") or 0.0) < config.FOLLOWUP_WINDOW_S
