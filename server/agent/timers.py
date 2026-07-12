"""Persisted background timers.

Timers are stored as wall-clock start_time + duration, not a running
asyncio.sleep — so state survives a process restart (e.g. Render's free
tier spinning the dyno down and back up). Progress and due-checks are pure
arithmetic; the only LLM call happens once, when a timer is found to have
completed.
"""

from __future__ import annotations
import json
import time
import uuid
from pathlib import Path

TIMERS_FILE = Path(__file__).parent.parent / "data" / "timers.json"


def _load() -> dict:
    if not TIMERS_FILE.exists():
        return {}
    try:
        return json.loads(TIMERS_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _save(data: dict):
    TIMERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMERS_FILE.write_text(json.dumps(data, indent=2))


def start_timer(label: str, duration_seconds: int, context: str = "") -> str:
    data = _load()
    timer_id = str(uuid.uuid4())[:8]
    data[timer_id] = {
        "label": label,
        "start_time": time.time(),
        "duration_seconds": duration_seconds,
        "context": context,
        "fired": False,
        "message": None,
    }
    _save(data)
    return timer_id


def due_unfired() -> list[dict]:
    """Timers whose duration has elapsed but haven't gotten a completion message yet."""
    data = _load()
    now = time.time()
    return [
        {"id": tid, **t}
        for tid, t in data.items()
        if not t["fired"] and now - t["start_time"] >= t["duration_seconds"]
    ]


def mark_fired(timer_id: str, message: str):
    data = _load()
    if timer_id in data:
        data[timer_id]["fired"] = True
        data[timer_id]["message"] = message
        _save(data)


def pop_completions() -> list[dict]:
    """Return fired timers (not yet delivered to a client) and remove them."""
    data = _load()
    completed = [{"id": tid, **t} for tid, t in data.items() if t["fired"]]
    for c in completed:
        del data[c["id"]]
    if completed:
        _save(data)
    return completed


def active_progress() -> list[dict]:
    """Pure-math progress snapshot for timers still running. No LLM cost."""
    data = _load()
    now = time.time()
    result = []
    for tid, t in data.items():
        if t["fired"]:
            continue
        elapsed = now - t["start_time"]
        duration = t["duration_seconds"]
        pct = min(100, int(elapsed / duration * 100)) if duration > 0 else 100
        result.append({
            "id": tid,
            "label": t["label"],
            "percent_done": pct,
            "remaining_seconds": max(0, int(duration - elapsed)),
        })
    return result
