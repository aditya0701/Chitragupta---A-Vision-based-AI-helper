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
