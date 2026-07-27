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
    """Timers whose duration has elapsed but haven't gotten a completion message yet.

    Excludes timers already claimed via mark_firing() — without this, two
    concurrent callers (the poll route and a live-frame chat turn both
    landing in the same window) could each pick up the same due timer and
    fire two completion messages for it.
    """
    data = _load()
    now = time.time()
    return [
        {"id": tid, **t}
        for tid, t in data.items()
        if not t["fired"] and not t.get("firing") and now - t["start_time"] >= t["duration_seconds"]
    ]


def mark_firing(timer_id: str):
    """Claim a timer before making the (slow, awaited) completion call for it."""
    data = _load()
    if timer_id in data:
        data[timer_id]["firing"] = True
        _save(data)


def mark_fired(timer_id: str, message: str, debug: dict | None = None):
    data = _load()
    if timer_id in data:
        data[timer_id]["fired"] = True
        data[timer_id]["firing"] = False
        data[timer_id]["message"] = message
        data[timer_id]["debug"] = debug
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


def cancel_timer(ref: str) -> str:
    """Cancel a timer by id or label (case-insensitive, exact then substring).

    Deletes outright rather than marking it fired: a cancelled timer must not
    produce a completion message, and mark_fired() is precisely what queues one
    for delivery. Already-fired-but-undelivered timers are cancellable too —
    "never mind" arriving a second before the poll picks it up should still
    suppress it, which only works if pop_completions() can no longer see it.
    """
    data = _load()
    ref_l = ref.strip().lower()
    matches = [tid for tid in data if tid == ref.strip()]
    if not matches:
        matches = [tid for tid, t in data.items() if t["label"].strip().lower() == ref_l]
    if not matches:
        matches = [tid for tid, t in data.items() if ref_l in t["label"].strip().lower()]

    if not matches:
        if not data:
            return "No timers are running, so there was nothing to cancel."
        running = ", ".join(f"'{t['label']}'" for t in data.values())
        return f"No timer matching '{ref}'. Currently running: {running}."
    if len(matches) > 1:
        # Ambiguous on purpose: cancelling the wrong timer is silent and
        # unrecoverable (the user finds out when it never goes off), so ask
        # rather than guess.
        labels = ", ".join(f"'{data[t]['label']}'" for t in matches)
        return (
            f"'{ref}' matches more than one timer ({labels}) — ask the user which "
            "one they mean, then call cancel_timer again with the exact label."
        )

    label = data[matches[0]]["label"]
    del data[matches[0]]
    _save(data)
    return f"Cancelled the timer for '{label}'. It will not go off."


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
