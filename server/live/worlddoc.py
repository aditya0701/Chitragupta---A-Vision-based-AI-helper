"""The world document — primary state of the live system.

Everything the model knows between calls lives here, in five sections:

  tasks         structured plan state (same shape as the old tasklist)
  expectations  things that *should* happen, each checkable — time-anchored
                ones by pure arithmetic, event-anchored ones by the model
                against the current frame
  environment   durable spatial/world facts ("chili is on the top shelf")
  narrative     compacted history — time-span summaries of old ticks
  recent        raw timestamped tick captions, bounded; overflow is
                compacted into narrative, not silently dropped

Every entry is timestamped (Qwen3-VL's textual-timestamp lesson applied at
the system level: temporal grounding should be *readable*, not inferred).
Persistence is a single JSON file so a Render restart loses nothing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from . import config
from ..config import settings

logger = logging.getLogger("chitragupt.live")

DOC_FILE = Path(__file__).parent.parent / "data" / "live" / "worlddoc.json"

VALID_TASK_STATUSES = {"pending", "in_progress", "completed", "skipped"}
VALID_EXPECTATION_STATUSES = {"open", "satisfied", "fired", "cancelled"}
VALID_ANCHORS = {"time", "event"}
VALID_PRIORITIES = {"high", "normal", "low"}

_lock = threading.Lock()  # file-level; async callers already serialize via LiveAgent._lock


def _now() -> float:
    return time.time()


_tz_cache: Optional[ZoneInfo] = None


def _tz() -> Optional[ZoneInfo]:
    """The user's zone, not the server's. Render runs UTC, so the naive
    datetime.fromtimestamp() this used to call stamped every timestamp in the
    doc — including the "[Current time]" header the model does all its temporal
    arithmetic against — two hours behind a Berlin user. Ask "will this be done
    by four?" and it was answering about a different four. Falls back to server
    local if the zone name is bad or tzdata is missing."""
    global _tz_cache
    if _tz_cache is None:
        try:
            _tz_cache = ZoneInfo(settings.DEFAULT_TIMEZONE)
        except Exception:
            logger.warning(
                "DEFAULT_TIMEZONE %r unusable — world-doc timestamps fall back to "
                "server local time", settings.DEFAULT_TIMEZONE,
            )
            return None
    return _tz_cache


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, _tz()).strftime("%H:%M:%S")


def _empty_doc() -> dict:
    return {
        "title": None,
        "session_started": _now(),
        "last_spoken_ts": 0.0,
        "last_user_turn_ts": 0.0,
        # Monotonic, bumped once per write window. Ticks and chat turns now run
        # concurrently, so their responses can arrive out of order — a tick that
        # started first can reply after a chat turn that started later, carrying
        # an older render. The client drops any render whose rev it has already
        # passed, which is the only thing stopping a stale doc from stamping
        # over a fresh one in the panel.
        "rev": 0,
        # The words of the last user turn, not just its clock time. Stage 2's
        # speech prompt used to get only `last_user_turn_ts`, so it was asked
        # "does this answer what they wanted?" while having no idea what they
        # had wanted. Observed live 2026-08-10: the camera reported "boneless
        # chicken breast is present and within reach", Stage 2 read that
        # sentence, could not connect it to a request it had never seen, judged
        # itself to be narrating, and stayed silent for 65s until the user
        # asked again. DECISIONS.md §6.4.
        "last_user_msg": "",
        "vision_focus": None,
        "proposal": None,
        "tasks": [],
        "expectations": [],
        "wanted": [],
        "environment": [],
        "narrative": [],
        "recent": [],
    }


def load() -> dict:
    with _lock:
        if not DOC_FILE.exists():
            return _empty_doc()
        try:
            doc = json.loads(DOC_FILE.read_text())
        except json.JSONDecodeError:
            logger.warning("worlddoc.json unreadable — starting fresh")
            return _empty_doc()
    # Backfill any missing sections so schema growth never crashes old files.
    empty = _empty_doc()
    for key, default in empty.items():
        doc.setdefault(key, default)
    return doc


def save(doc: dict):
    with _lock:
        DOC_FILE.parent.mkdir(parents=True, exist_ok=True)
        DOC_FILE.write_text(json.dumps(doc, indent=2))


def clear():
    with _lock:
        if DOC_FILE.exists():
            DOC_FILE.unlink()


# ── Tasks ────────────────────────────────────────────────────────────────────

def set_tasks(doc: dict, title: str, items: list[dict]) -> str:
    """Full-replace, TodoWrite-style, with the same defenses the old
    tasklist earned the hard way: content-key aliases and a refusal to wipe
    a populated list when every incoming item fails to parse."""
    existing_ids = {t["content"]: t["id"] for t in doc["tasks"]}
    normalized = []
    for item in items or []:
        content = (item.get("content") or item.get("task") or item.get("label") or "").strip()
        if not content:
            continue
        status = item.get("status", "pending")
        if status not in VALID_TASK_STATUSES:
            status = "pending"
        normalized.append({
            "id": item.get("id") or existing_ids.get(content) or uuid.uuid4().hex[:8],
            "content": content,
            "status": status,
            "note": item.get("note"),
            "last_mention_ts": _now(),
        })
    if items and not normalized and doc["tasks"]:
        logger.warning("set_tasks: all incoming items unparseable — keeping existing tasks")
        return "No usable items received — existing task list kept unchanged."
    doc["title"] = title or doc["title"]
    doc["tasks"] = normalized
    return f"Task list '{doc['title']}' updated ({len(normalized)} items)."


# ── Proposed plans ───────────────────────────────────────────────────────────
#
# The plan the assistant WANTS to commit, held out of `tasks` until the user
# says yes. This exists because of the asymmetry between the two kinds of thing
# the model writes.
#
# Observations — a caption, an environment fact, a resolved watch — are reports
# of what it saw. They should be written silently and immediately; gating them
# on approval would turn every tick into a permission prompt and destroy the
# whole point of a hands-free assistant.
#
# A PLAN is different. It is a decision about what the user is going to spend
# the next hour doing, made from a photograph and a web search, and once it
# lands in `tasks` it is re-injected into every subsequent prompt as settled
# fact. The model then reads its own guess back as memory and holds the user to
# it — the same "a wrong record survives correction" failure that
# retract_environment_fact exists for, except a plan is much harder to unpick
# because tasks, expectations and vision focus all get built on top of it.
#
# So: propose out loud, commit on assent. One proposal at a time, replaced
# rather than appended (same reasoning as set_vision_focus — a second proposal
# means the model changed its mind, not that there are now two plans).

def propose_plan(doc: dict, title: str, steps: list) -> str:
    """Put a plan up for the user's approval WITHOUT touching `tasks`."""
    normalized = []
    for item in steps or []:
        if isinstance(item, dict):
            content = (item.get("content") or item.get("task")
                       or item.get("label") or "").strip()
            note = (item.get("note") or "").strip() or None
        else:
            content, note = str(item).strip(), None
        if content:
            normalized.append({"content": content, "note": note})

    if not normalized:
        return "A proposal needs at least one step — nothing was proposed."

    doc["proposal"] = {
        "id": uuid.uuid4().hex[:8],
        "title": (title or "").strip() or "Plan",
        "steps": normalized,
        "ts": _now(),
        # When the user was last actually ASKED about it, which is not the same
        # as when it was proposed: the re-raise trigger measures from here so a
        # plan mentioned again on a tick doesn't get nagged about on a timer.
        "raised_ts": _now(),
    }
    return (
        f"Plan PROPOSED ({len(normalized)} steps) — NOT committed, and NOT in the task "
        f"list. Nothing is tracking it and no expectations exist for it yet. Say it out "
        f"loud now, briefly, and ask the user to confirm. Call commit_plan when they "
        f"agree; propose_plan again if they want it changed."
    )


def get_proposal(doc: dict) -> Optional[dict]:
    return doc.get("proposal") or None


def commit_proposal(doc: dict, note: str = "") -> str:
    """Promote the pending proposal into the real task list."""
    proposal = doc.get("proposal")
    if not proposal:
        return ("No plan is pending approval — nothing to commit. If you want to write a "
                "plan the user has already agreed to, use update_tasks.")
    items = [{"content": s["content"], "status": "pending", "note": s.get("note")}
             for s in proposal["steps"]]
    result = set_tasks(doc, proposal["title"], items)
    doc["proposal"] = None
    return (f"Plan committed — {result} Now set expectations for the steps that need "
            f"them, and mark_task the first one in_progress when they start it."
            + (f" Note: {note}" if note else ""))


def discard_proposal(doc: dict, reason: str = "") -> str:
    proposal = doc.get("proposal")
    if not proposal:
        return "No plan is pending approval."
    doc["proposal"] = None
    return (f"Proposed plan '{proposal['title']}' discarded."
            + (f" ({reason})" if reason else ""))


def mark_proposal_raised(doc: dict):
    """Record that the user has just been asked about the pending plan."""
    if doc.get("proposal"):
        doc["proposal"]["raised_ts"] = _now()


def find_task(doc: dict, ref: str) -> Optional[dict]:
    ref_l = (ref or "").strip().lower()
    return next(
        (t for t in doc["tasks"] if t["id"] == ref or t["content"].lower() == ref_l),
        None,
    )


def touch_task(doc: dict, ref: str):
    """Mark a task as recently-mentioned so the staleness trigger resets."""
    task = find_task(doc, ref)
    if task:
        task["last_mention_ts"] = _now()


# ── Expectations ─────────────────────────────────────────────────────────────

def add_expectation(
    doc: dict,
    description: str,
    anchor: str,
    due_in_seconds: Optional[float] = None,
    condition: Optional[str] = None,
    priority: str = "normal",
    task_ref: Optional[str] = None,
) -> str:
    description = (description or "").strip()
    if not description:
        return "Expectation needs a description."
    if anchor not in VALID_ANCHORS:
        return f"anchor must be one of {sorted(VALID_ANCHORS)}."
    if anchor == "time" and not due_in_seconds:
        return "A time-anchored expectation needs due_in_seconds."
    if anchor == "event" and not (condition or "").strip():
        return "An event-anchored expectation needs a condition (what to watch for)."
    if priority not in VALID_PRIORITIES:
        priority = "normal"

    # No duplicate detection here, deliberately. A Jaccard-overlap check was
    # tried and removed: it failed in both directions. The duplicates that
    # actually occurred were rephrasings sharing only ~19% of their words ("Car
    # is safe to work under" vs "The car should be settled firmly on both
    # ramps"), well under any usable threshold — while a threshold low enough
    # to catch them merged genuinely distinct watches that differed only in the
    # object ("is the 10mm socket seated" vs "the 12mm"). Silently dropping a
    # watch the user is relying on is far worse than carrying a duplicate.
    #
    # The duplicate pressure it was built for is gone anyway: form and safety
    # moved to set_vision_focus, which replaces rather than appends, and
    # _vision_questions caps how many reach the camera per frame.
    task = find_task(doc, task_ref) if task_ref else None
    exp = {
        "id": uuid.uuid4().hex[:8],
        "description": description,
        "anchor": anchor,
        # Absolute wall-clock deadline, same restart-resilience lesson as
        # the old timers: never store a countdown.
        "due_ts": (_now() + float(due_in_seconds)) if anchor == "time" else None,
        "condition": (condition or "").strip() or None,
        "priority": priority,
        "status": "open",
        "created_ts": _now(),
        "task_id": task["id"] if task else None,
    }
    doc["expectations"].append(exp)
    if anchor == "time":
        return f"Expectation set: '{description}' due at {fmt_ts(exp['due_ts'])} (id {exp['id']})."
    return f"Expectation set: '{description}' — watching for: {exp['condition']} (id {exp['id']})."


def find_expectation(doc: dict, ref: str) -> Optional[dict]:
    ref_l = (ref or "").strip().lower()
    return next(
        (e for e in doc["expectations"]
         if e["id"] == ref or e["description"].lower() == ref_l),
        None,
    )


def resolve_expectation(doc: dict, ref: str, outcome: str = "satisfied", note: str = "") -> str:
    exp = find_expectation(doc, ref)
    if not exp:
        return f"No expectation matching '{ref}'."
    if outcome not in ("satisfied", "cancelled"):
        outcome = "satisfied"
    exp["status"] = outcome
    exp["resolved_ts"] = _now()
    if note:
        exp["note"] = note
    if exp.get("task_id"):
        touch_task(doc, exp["task_id"])
    return f"Expectation '{exp['description']}' marked {outcome}."


def open_expectations(doc: dict) -> list[dict]:
    return [e for e in doc["expectations"] if e["status"] == "open"]


# ── The find list ────────────────────────────────────────────────────────────
#
# Objects the user asked to have found. The camera is asked about all of them
# by name on every frame, and a labelled FOUND answer — not a model's reading
# of a description — is the only thing that can tick one off.
#
# THERE IS DELIBERATELY NO TOOL THAT MARKS AN ITEM FOUND. `add_wanted` opens a
# search and `drop_wanted` cancels one; nothing lets the reasoning model write
# status="found". Only fold_wanted() can, and it does it by string-matching an
# explicit answer produced by the stage that actually saw the pixels.
#
# That asymmetry is the whole point. The failure it prevents is on record: the
# user wanted black-eyed beans, the caption said "several bags of lentils", and
# the reasoning model — with no answer to read — upgraded that into "I can see
# the beans". An inference stood in for an observation. Under this design an
# inference cannot reach the found state, because the found state has no
# model-facing door.


def add_wanted(doc: dict, items: list[str], because: str = "") -> str:
    """Open a search for one or more objects. Takes a list, so "find the
    chicken and the onions" is one call rather than two."""
    names = [str(i).strip() for i in (items or []) if str(i).strip()]
    if not names:
        return "add_wanted needs at least one item to look for."

    existing = {w["item"].lower() for w in doc["wanted"] if w["status"] == "open"}
    added, dupes = [], []
    for name in names:
        if name.lower() in existing:
            dupes.append(name)
            continue
        doc["wanted"].append({
            "id": uuid.uuid4().hex[:8],
            "item": name,
            "asked_ts": _now(),
            "asked_text": (because or "").strip(),
            "status": "open",
            "found_ts": None,
            "where": None,
            "misses": 0,
            "unclears": 0,
            # Kept separate from status="found" on purpose. `found` is a fact
            # about the world; `announced` is a fact about speech. If the
            # utterance fails — the model returns [SILENT], the network drops,
            # the tick dies — the item is still found and the user is still
            # owed it, and fusing the two would make a failed announcement
            # indistinguishable from a delivered one. Because the announce
            # trigger tests doc state rather than this tick's caption, it
            # simply re-fires next tick until the words land.
            "announced": False,
            "stuck_raised": False,
        })
        existing.add(name.lower())
        added.append(name)

    # Cap on OPEN items only — a found item costs the camera nothing.
    open_now = [w for w in doc["wanted"] if w["status"] == "open"]
    if len(open_now) > config.MAX_WANTED:
        for stale in open_now[: len(open_now) - config.MAX_WANTED]:
            stale["status"] = "dropped"

    if not added:
        return f"Already looking for: {', '.join(dupes)}. The camera is asked every frame."
    msg = f"Now looking for on every frame: {', '.join(added)}."
    if dupes:
        msg += f" (Already watching for {', '.join(dupes)}.)"
    return msg + " You will be told the moment any of them is seen — do not promise to look, you already are."


def find_wanted(doc: dict, ref: str) -> Optional[dict]:
    ref_l = (ref or "").strip().lower()
    if not ref_l:
        return None
    return next(
        (w for w in doc["wanted"]
         if w["id"] == ref or w["item"].lower() == ref_l or ref_l in w["item"].lower()),
        None,
    )


def drop_wanted(doc: dict, ref: str) -> str:
    item = find_wanted(doc, ref)
    if not item:
        return f"Not looking for anything matching '{ref}'."
    item["status"] = "dropped"
    return f"Stopped looking for {item['item']}."


def open_wanted(doc: dict) -> list[dict]:
    return [w for w in doc.get("wanted", []) if w["status"] == "open"]


def wanted_names(doc: dict) -> list[str]:
    """The item names put to the camera this frame, in one question."""
    return [w["item"] for w in open_wanted(doc)][: config.MAX_WANTED]


_VERDICTS = ("FOUND", "NOT VISIBLE", "UNCLEAR")


def _verdict_re(item: str) -> re.Pattern:
    # Anchored at line start and requiring the item name followed by a
    # separator and a verdict token, so the word "onions" appearing in the
    # prose description below the answers can never be mistaken for an answer
    # about onions.
    return re.compile(
        r"^\s*[-*\u2022]?\s*" + re.escape(item) + r"\s*[:\-\u2013\u2014]+\s*"
        r"(FOUND|NOT[ _]VISIBLE|UNCLEAR)\b[\s:\-\u2013\u2014]*(.*)$",
        re.IGNORECASE,
    )


def fold_wanted(doc: dict, caption: str) -> list[dict]:
    """Match the camera's labelled answers against the open find list.

    Pure string matching, zero tokens, no judgement. Returns the items newly
    moved to `found` so the caller can promote their locations into durable
    environment facts.

    Matching is BY NAME, never by position. An earlier draft numbered the
    questions Q1/Q2/Q3 and mapped answers back by index — but that list is
    rebuilt every tick from open state, priority-sorted, while the vision call
    runs unlocked for ~1.5s. Close one watch or add a high-priority safety one
    mid-search and every index shifts, so `Q2: FOUND` would be attributed to
    the wrong object: right location, wrong item, stated as fact, and then
    written into durable memory. Names have no such failure mode.
    """
    text = caption or ""
    if not text.strip():
        return []
    lines = text.splitlines()
    newly_found = []

    for item in open_wanted(doc):
        pattern = _verdict_re(item["item"])
        match = next((m for m in (pattern.match(ln) for ln in lines) if m), None)
        if not match:
            # Absent from the answer block is NOT a miss. A truncated reply or
            # a model that answered only some items must never be able to drive
            # a give-up: silence is not evidence, which is the same rule that
            # makes NOT VISIBLE an explicit result rather than an omission.
            continue
        verdict = match.group(1).upper().replace("_", " ")
        detail = (match.group(2) or "").strip()
        if verdict == "FOUND":
            # Claims the item, exactly as triggers.check claims a fired
            # expectation: this caption sits in `recent` for another 24 frames
            # and must not be able to re-fire the announcement.
            item["status"] = "found"
            item["found_ts"] = _now()
            item["where"] = detail or "seen in frame, no location given"
            newly_found.append(item)
        elif verdict == "NOT VISIBLE":
            item["misses"] = int(item.get("misses") or 0) + 1
        elif verdict == "UNCLEAR":
            item["unclears"] = int(item.get("unclears") or 0) + 1

    return newly_found


def unannounced_finds(doc: dict) -> list[dict]:
    """Found, but the user has not actually been told yet."""
    return [w for w in doc.get("wanted", [])
            if w["status"] == "found" and not w.get("announced")]


def mark_wanted_announced(doc: dict, items: list[dict]):
    """Called only after speech has actually survived the gate."""
    ids = {w["id"] for w in items}
    for w in doc.get("wanted", []):
        if w["id"] in ids:
            w["announced"] = True


# ── Vision focus ─────────────────────────────────────────────────────────────

VALID_FOCUS_MODES = {"form", "read"}


def set_vision_focus(doc: dict, brief: str, detail: str = "fine",
                     mode: str = "form") -> str:
    """The standing lens the camera looks through — one brief, replaced not
    appended.

    The counterpart to event-anchored watches, and better than them for form
    and safety. A watch is a closed question with a definite answer, which is
    right for a search ("is the lobhiya bag visible?") and wrong for technique:
    we cannot know in advance what the camera will see, so a checklist written
    before the frame arrives is guesswork, and anything not on it is invisible.

    A brief instead says what the job is and which dimensions matter, and
    leaves room to report the unanticipated — the thing a list can never do.

    Replace-not-append is the whole point. Nine form watches accumulated on one
    oil-filter plan, five of them restatements, because every planning pass
    could add more and nothing could merge them. There is exactly one focus, so
    duplicates are impossible by construction.
    """
    brief = (brief or "").strip()
    if not brief:
        doc["vision_focus"] = None
        return "Vision focus cleared — the camera goes back to plain description, coarse frames."
    if mode not in VALID_FOCUS_MODES:
        mode = "form"
    if detail not in ("fine", "coarse"):
        detail = "fine"
    # Reading is impossible at 640px — that is the entire reason the fine tier
    # exists (CLAUDE.md: resolution discarded in the browser cannot be
    # recovered). Asking to read at coarse would silently return "text is not
    # legible" forever, so the mode wins over the argument.
    if mode == "read":
        detail = "fine"
    prev = doc.get("vision_focus") or {}
    doc["vision_focus"] = {
        "brief": brief,
        "detail": detail,
        "mode": mode,
        "ts": _now(),
        # Frames spent at fine on THIS focus. Carried over ONLY when the focus
        # is re-sent completely unchanged, which is a no-op re-assertion —
        # otherwise the model could dodge the cap forever by repeating itself.
        # Any real change (new brief, or a different detail level) is new
        # intent and earns a fresh budget, so a step that legitimately needs
        # another close look is never locked out by an earlier one.
        "fine_frames": (prev.get("fine_frames", 0)
                        if prev.get("detail") == detail and prev.get("brief") == brief
                        and prev.get("mode") == mode else 0),
    }
    return (f"Camera focus set ({mode}, {detail} frames): {brief}"
            + ("" if detail == "fine" else
               " — call again with detail='fine' when you need to see small things."))


def get_vision_focus(doc: dict) -> Optional[str]:
    focus = doc.get("vision_focus")
    return (focus or {}).get("brief") or None


def focus_detail(doc: dict) -> Optional[str]:
    """The resolution this focus asked for, or None if no focus is set."""
    focus = doc.get("vision_focus")
    return (focus or {}).get("detail") if focus else None


def focus_mode(doc: dict) -> Optional[str]:
    """'form' (watch how the work is done) or 'read' (transcribe text)."""
    focus = doc.get("vision_focus")
    return (focus or {}).get("mode", "form") if focus else None


def charge_focus_frame(doc: dict):
    """Count one frame captured at fine against the active focus."""
    focus = doc.get("vision_focus")
    if focus and focus.get("detail") == "fine":
        focus["fine_frames"] = int(focus.get("fine_frames") or 0) + 1


# ── Environment facts & recent captions ──────────────────────────────────────

def add_environment_fact(doc: dict, fact: str) -> str:
    fact = (fact or "").strip()
    if not fact:
        return "Empty fact ignored."
    doc["environment"].append({"ts": _now(), "fact": fact})
    if len(doc["environment"]) > config.MAX_ENV_FACTS:
        del doc["environment"][: len(doc["environment"]) - config.MAX_ENV_FACTS]
    return f"Noted: {fact}"


def retract_environment_fact(doc: dict, fact_match: str, correction: str = "") -> str:
    """Remove environment facts containing `fact_match`, optionally replacing
    them with a correction in the same call.

    The counterpart to add_environment_fact, which is append-only. v1 learned
    this the hard way and v2 shipped without it — see agent/tasklist.py's
    retract_observation: the model logged "found the toor dal" for what was
    actually a bag of black-eyed beans, the user corrected it, and nothing
    could act on the correction. The wrong fact rode along in every subsequent
    prompt injection and the model repeated it for four turns.

    **A false fact that survives an explicit correction is worse than no fact
    at all**, because once logged it is indistinguishable from a verified one.

    Two v2-specific details:

    `correction` exists because deleting alone is not enough here. The raw
    captions that produced the wrong inference are still sitting in `recent`
    and will keep suggesting it — the model can re-derive the same claim on the
    very next tick from the very same evidence. A durable "the bag on the
    pantry shelf is black-eyed beans, NOT toor dal" actively blocks that, where
    a hole in the fact list does not.

    Substring matching rather than an index, case-insensitive, removing every
    match: the model works from the rendered [Known environment facts] text, so
    it can quote a fragment far more reliably than it can count positions, and
    a wrong fact tends to have been logged more than once across ticks.
    """
    needle = (fact_match or "").strip().lower()
    if not needle:
        return "retract_environment_fact needs the text of the fact to remove."

    before = len(doc["environment"])
    doc["environment"] = [f for f in doc["environment"] if needle not in f["fact"].lower()]
    removed = before - len(doc["environment"])

    note = ""
    if correction and correction.strip():
        add_environment_fact(doc, correction.strip())
        note = f" Recorded instead: {correction.strip()}"

    if not removed:
        return (f"No environment fact matching '{fact_match}' — check the "
                f"[Known environment facts] text.{note}")
    return f"Removed {removed} environment fact(s) matching '{fact_match}'.{note}"


def add_recent(doc: dict, caption: str) -> list[dict]:
    """Append a raw tick caption. Returns the batch that should be compacted
    (oldest entries beyond the bound), already removed from `recent` — the
    caller owns getting them summarized into `narrative`. Never silently
    drops raw captions."""
    doc["recent"].append({"ts": _now(), "text": (caption or "").strip()})
    if len(doc["recent"]) <= config.RECENT_MAX:
        return []
    batch = doc["recent"][: config.COMPACT_BATCH]
    doc["recent"] = doc["recent"][config.COMPACT_BATCH:]
    return batch


def add_narrative(doc: dict, start_ts: float, end_ts: float, text: str):
    doc["narrative"].append({"start_ts": start_ts, "end_ts": end_ts, "text": (text or "").strip()})
    if len(doc["narrative"]) > config.MAX_NARRATIVE:
        del doc["narrative"][: len(doc["narrative"]) - config.MAX_NARRATIVE]


def last_caption(doc: dict) -> Optional[str]:
    return doc["recent"][-1]["text"] if doc["recent"] else None


# ── Rendering ────────────────────────────────────────────────────────────────

_TASK_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]", "skipped": "[-]"}


def render(doc: dict, recent_limit: Optional[int] = None) -> str:
    """The doc as prompt text. Header carries the current wall-clock time so
    every temporal comparison is readable arithmetic for the model.

    Section order is stability-first (title/tasks/narrative/environment
    change rarely, recent changes every tick) so DeepSeek's prefix cache
    gets the longest possible unchanged prefix across consecutive ticks.
    """
    now = _now()
    lines = [f"[Current time: {fmt_ts(now)}]"]

    if doc.get("title"):
        lines.append(f"\n[Goal] {doc['title']}")

    if doc["tasks"]:
        lines.append("\n[Tasks]")
        for t in doc["tasks"]:
            mark = _TASK_MARKS.get(t["status"], "[ ]")
            line = f"{mark} {t['content']}"
            if t.get("note"):
                line += f"  ({t['note']})"
            lines.append(line)

    # Rendered right after the tasks it is NOT part of, and labelled hard.
    # A proposal that reads like a task list is worse than no proposal at all —
    # the model would act on it as though the user had agreed.
    proposal = doc.get("proposal")
    if proposal:
        lines.append(f"\n[PROPOSED PLAN — NOT COMMITTED, proposed {fmt_ts(proposal['ts'])}]")
        lines.append(f"Goal: {proposal['title']}")
        for i, step in enumerate(proposal["steps"], 1):
            line = f"  {i}. {step['content']}"
            if step.get("note"):
                line += f"  ({step['note']})"
            lines.append(line)
        lines.append(
            "The user has NOT agreed to this. It is not the task list, nothing is "
            "tracking it, and it has no expectations. If they have since agreed, call "
            "commit_plan. If they changed something, call propose_plan again with the "
            "change. If they said no or moved on, call discard_plan. Do not start "
            "working through these steps and do not treat them as decided."
        )

    # The model must be able to see its own standing focus, or it cannot know
    # whether to update it, and cannot tell that it left the camera on fine.
    focus = doc.get("vision_focus")
    if focus:
        spent = int(focus.get("fine_frames") or 0)
        lines.append(f"\n[Camera focus — {focus.get('detail', 'fine')} frames"
                     + (f", {spent} used" if spent else "") + "]")
        lines.append(focus["brief"])

    opens = open_expectations(doc)
    if opens:
        lines.append("\n[Open expectations]")
        for e in opens:
            if e["anchor"] == "time":
                remaining = e["due_ts"] - now
                when = (f"due in {int(remaining // 60)}m{int(remaining % 60):02d}s"
                        if remaining > 0 else f"OVERDUE by {int(-remaining // 60)}m{int(-remaining % 60):02d}s")
                lines.append(f"- ({e['id']}, {e['priority']}) {e['description']} — {when}")
            else:
                lines.append(f"- ({e['id']}, {e['priority']}) {e['description']} — fires when: {e['condition']}")

    # Between expectations and the narrative: it changes more often than tasks
    # and less often than `recent`, which keeps the stability-first ordering
    # (and so the prefix cache) intact.
    wanted = [w for w in doc.get("wanted", []) if w["status"] in ("open", "found")]
    if wanted:
        lines.append("\n[Looking for — the camera is asked about these by name every frame]")
        for w in wanted:
            if w["status"] == "found":
                told = "already told the user" if w.get("announced") else "NOT YET TOLD"
                lines.append(f"- {w['item']} — FOUND {fmt_ts(w['found_ts'])}: "
                             f"{w['where']}  ({told})")
            else:
                seen = []
                if w.get("misses"):
                    seen.append(f"not visible in {w['misses']} frames")
                if w.get("unclears"):
                    seen.append(f"unclear in {w['unclears']}")
                detail = "; ".join(seen) or "just started looking"
                lines.append(f"- {w['item']} — still looking "
                             f"(asked {fmt_ts(w['asked_ts'])}, {detail})")
        lines.append(
            "You do not need to do anything to search for these — the camera is "
            "already asked about every one of them on every frame, and you will be "
            "told the moment one is seen. Never say you will 'keep an eye out' as if "
            "it were a future action."
        )

    if doc["narrative"]:
        lines.append("\n[Earlier this session]")
        for n in doc["narrative"]:
            lines.append(f"- {fmt_ts(n['start_ts'])}–{fmt_ts(n['end_ts'])}: {n['text']}")

    if doc["environment"]:
        lines.append("\n[Known environment facts]")
        for f in doc["environment"]:
            lines.append(f"- ({fmt_ts(f['ts'])}) {f['fact']}")

    recent = doc["recent"]
    if recent_limit is not None:
        recent = recent[-recent_limit:]
    if recent:
        lines.append("\n[Recent observations]")
        for r in recent:
            lines.append(f"- {fmt_ts(r['ts'])}: {r['text']}")

    return "\n".join(lines)
