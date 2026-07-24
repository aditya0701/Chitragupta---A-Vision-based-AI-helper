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
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger("chitragupt.live")

DOC_FILE = Path(__file__).parent.parent / "data" / "live" / "worlddoc.json"

VALID_TASK_STATUSES = {"pending", "in_progress", "completed", "skipped"}
VALID_EXPECTATION_STATUSES = {"open", "satisfied", "fired", "cancelled"}
VALID_ANCHORS = {"time", "event"}
VALID_PRIORITIES = {"high", "normal", "low"}

_lock = threading.Lock()  # file-level; async callers already serialize via LiveAgent._lock


def _now() -> float:
    return time.time()


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _empty_doc() -> dict:
    return {
        "title": None,
        "session_started": _now(),
        "last_spoken_ts": 0.0,
        "tasks": [],
        "expectations": [],
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


# ── Environment facts & recent captions ──────────────────────────────────────

def add_environment_fact(doc: dict, fact: str) -> str:
    fact = (fact or "").strip()
    if not fact:
        return "Empty fact ignored."
    doc["environment"].append({"ts": _now(), "fact": fact})
    if len(doc["environment"]) > config.MAX_ENV_FACTS:
        del doc["environment"][: len(doc["environment"]) - config.MAX_ENV_FACTS]
    return f"Noted: {fact}"


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
