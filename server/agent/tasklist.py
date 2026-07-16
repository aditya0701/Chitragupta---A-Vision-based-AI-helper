"""Persisted task/document tracking.

A structured living document the model reads and rewrites each turn, instead
of relying on raw conversation history for "what's done, what's left, what
got substituted." Modeled on Claude Code's own TodoWrite tool: the model
sends the *entire* updated list every time it changes anything — no separate
add/remove/branch tools, no diffing logic, the server just persists whatever
it's given. Completed items stay in the list (marked, not deleted) so the
document doubles as a record of what happened.
"""

from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chitragupt")

DOCUMENT_FILE = Path(__file__).parent.parent / "data" / "document.json"

VALID_STATUSES = {"pending", "in_progress", "completed", "skipped"}


def get_document() -> Optional[dict]:
    if not DOCUMENT_FILE.exists():
        return None
    try:
        return json.loads(DOCUMENT_FILE.read_text())
    except json.JSONDecodeError:
        return None


def set_document(title: str, items: list[dict]) -> dict:
    """Replace the whole document.

    Each item: {content, status, note?}. Ids are preserved across edits by
    matching on content against the previous version, so the same logical
    item keeps its identity as its status changes turn to turn.
    """
    existing = get_document() or {}
    existing_ids = {i.get("content"): i.get("id") for i in existing.get("items", [])}

    normalized = []
    for item in items:
        # The model occasionally uses "task" or "label" instead of the
        # documented "content" key (nothing structurally enforces the exact
        # field name — tool calls here are parsed from free-form JSON in the
        # response text, not a validated function-calling schema). Accept
        # the common variants rather than silently dropping the item.
        content = (item.get("content") or item.get("task") or item.get("label") or "").strip()
        if not content:
            continue
        status = item.get("status", "pending")
        if status not in VALID_STATUSES:
            status = "pending"
        item_id = item.get("id") or existing_ids.get(content) or str(uuid.uuid4())[:8]
        existing_item = next((i for i in existing.get("items", []) if i.get("id") == item_id), None)
        normalized.append({
            "id": item_id,
            "content": content,
            "status": status,
            "note": item.get("note"),
            # Preserved across edits like id/content — the model rewrites
            # the item list on every update_task_list call but doesn't know
            # about (and shouldn't have to resend) prior observations.
            "observations": (existing_item or {}).get("observations", []),
        })

    if items and not normalized and existing.get("items"):
        # Every incoming item failed to parse (e.g. a field-name mismatch
        # the alias handling above didn't catch) — refuse to silently
        # replace a populated document with an empty one. update_task_list
        # is a full-replace API, so a single malformed call would otherwise
        # wipe all prior task-list progress with no visible error.
        logger.warning(
            f"update_task_list: all {len(items)} incoming item(s) had no "
            "usable content field — keeping existing document unchanged."
        )
        return existing

    document = {"title": title, "items": normalized}
    DOCUMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_FILE.write_text(json.dumps(document, indent=2))
    return document


def clear_document():
    if DOCUMENT_FILE.exists():
        DOCUMENT_FILE.unlink()


def start_find_task(target: str) -> dict:
    """Register a "find <target>" item as the active goal — used by the
    request_live_search tool. Deliberately atomic and self-contained (adds
    one item to whatever document already exists, or creates a new one)
    rather than relying on the model separately calling update_task_list
    with the full list — a live-search request should always succeed at
    registering its goal, not depend on the model remembering a second step.
    """
    existing = get_document()
    content = f"Find {target.strip()}"
    if existing and existing.get("items"):
        # Don't duplicate if this exact search is already tracked.
        if any(i["content"] == content for i in existing["items"]):
            return existing
        return set_document(existing["title"], existing["items"] + [{"content": content, "status": "in_progress"}])
    return set_document(content, [{"content": content, "status": "in_progress"}])


MAX_OBSERVATIONS_PER_ITEM = 5


def add_observation(item_ref: str, note: str, found: bool = False) -> str:
    """Append a short note to whichever task-list item `item_ref` matches
    (by id or, case-insensitively, by content). Capped per item so the
    prompt injection in render_summary stays small regardless of session
    length — older notes are dropped, not the whole log.

    `found=True` also marks the matched item "completed" — a find-task
    (registered by request_live_search/start_find_task as "in_progress") is
    by definition done once the target's been spotted, so this is what lets
    agent.py detect "the active goal just finished" from task-list state
    alone, without a separate signal threaded through just for this.
    """
    document = get_document()
    if not document or not document.get("items"):
        return "No active task list — nothing to log this observation against."

    match = next(
        (i for i in document["items"]
         if i["id"] == item_ref or i["content"].lower() == item_ref.strip().lower()),
        None,
    )
    if not match:
        return f"No task list item matching '{item_ref}' — check the [Task list] content exactly."

    obs = match.setdefault("observations", [])
    obs.append(note.strip())
    if len(obs) > MAX_OBSERVATIONS_PER_ITEM:
        del obs[: len(obs) - MAX_OBSERVATIONS_PER_ITEM]

    if found:
        match["status"] = "completed"

    DOCUMENT_FILE.write_text(json.dumps(document, indent=2))
    return f"Logged observation for '{match['content']}'."


_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]", "skipped": "[-]"}


def render_summary(document: Optional[dict], lean: bool = False, observations: bool = True) -> str:
    """Render the document as compact text for prompt context.

    lean=True drops observations from completed/skipped items only — their
    history is a settled record, not something a live tick needs to
    re-check relevance against, so it doesn't need to keep riding along on
    every future tick. Without this, a long multi-item session regrows the
    exact prompt bulk that trimming conversation_history/native_tools was
    meant to remove, just from a different source. Used for is_live_frame
    calls.

    observations=False drops every item's observations outright (titles/
    status only) — a harder cut used only as an emergency degrade when a
    provider rejects a request as too large even after the lean trim.
    """
    if not document or not document.get("items"):
        return ""
    lines = [f"Task: {document['title']}"]
    for item in document["items"]:
        mark = _STATUS_MARKS.get(item["status"], "[ ]")
        line = f"{mark} {item['content']}"
        if item.get("note"):
            line += f"  ({item['note']})"
        lines.append(line)
        if not observations or (lean and item["status"] in ("completed", "skipped")):
            continue
        for obs in item.get("observations", []):
            lines.append(f"    - {obs}")
    return "\n".join(lines)
