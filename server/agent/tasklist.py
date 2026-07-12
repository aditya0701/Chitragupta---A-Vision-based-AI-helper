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
import uuid
from pathlib import Path
from typing import Optional

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
        content = (item.get("content") or "").strip()
        if not content:
            continue
        status = item.get("status", "pending")
        if status not in VALID_STATUSES:
            status = "pending"
        item_id = item.get("id") or existing_ids.get(content) or str(uuid.uuid4())[:8]
        normalized.append({
            "id": item_id,
            "content": content,
            "status": status,
            "note": item.get("note"),
        })

    document = {"title": title, "items": normalized}
    DOCUMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_FILE.write_text(json.dumps(document, indent=2))
    return document


def clear_document():
    if DOCUMENT_FILE.exists():
        DOCUMENT_FILE.unlink()


_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]", "skipped": "[-]"}


def render_summary(document: Optional[dict]) -> str:
    """Render the document as compact text for prompt context."""
    if not document or not document.get("items"):
        return ""
    lines = [f"Task: {document['title']}"]
    for item in document["items"]:
        mark = _STATUS_MARKS.get(item["status"], "[ ]")
        line = f"{mark} {item['content']}"
        if item.get("note"):
            line += f"  ({item['note']})"
        lines.append(line)
    return "\n".join(lines)
