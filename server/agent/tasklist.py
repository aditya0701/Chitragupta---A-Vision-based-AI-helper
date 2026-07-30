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

# How closely the vision stage has to look. "coarse" is the 640px live-tick
# default — fine for "is the thing there / what is happening". "fine" sends the
# frame at full resolution and lets the vision model read text, which is the
# only way label reading works: you cannot recover detail from a frame that was
# already downscaled before it left the browser.
VALID_DETAIL = {"coarse", "fine"}

# Hard ceiling on how many frames one item may be looked at closely for.
#
# The scope bound ("completing the step ends the close look") was the only
# control, and it did not hold: on 2026-07-28 the model set detail:"fine" once,
# never set it back despite the schema telling it to, the step stayed
# in_progress for the rest of the session, and every subsequent tick ran at
# full resolution. The session died at 198,310 of 200,000 daily Groq tokens.
#
# The lesson is general: a cost control that depends on the model's judgement
# is not a cost control. This is the arithmetic backstop that does not.
# 8 frames is enough for a genuine close look at a label (the real session read
# one in 2-3) while capping the damage at roughly 8 x 3424 rather than
# unbounded. See DECISIONS.md §6.4.
MAX_FINE_FRAMES_PER_ITEM = 8


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

        incoming_detail = (item.get("detail") or "").strip().lower()
        prev_detail = (existing_item or {}).get("detail")
        prev_used = (existing_item or {}).get("fine_frames_used", 0)
        prev_spent = (existing_item or {}).get("fine_budget_spent", False)
        detail = (incoming_detail if incoming_detail in VALID_DETAIL
                  else prev_detail or None)

        # Budget bookkeeping across a full replace. update_task_list resends
        # every item on every call, so this runs constantly and is exactly
        # where a naive rule leaks.
        #
        # The leak, found by the test: record_fine_frame() writes detail back
        # to "coarse" on exhaustion, so the model's very next full replace —
        # still carrying detail:"fine" because it hasn't noticed — looked like
        # a coarse->fine transition and refilled the budget. That is the same
        # drift the budget exists to stop, so exhaustion is recorded
        # explicitly and only the model standing the look down clears it.
        if incoming_detail == "coarse":
            # A deliberate stand-down. Clears exhaustion, so a later explicit
            # "fine" is a genuine second request and gets a fresh budget.
            fine_used, spent = 0, False
        elif prev_spent:
            # Budget already spent and never stood down: ignore the renewed
            # request entirely, including forcing detail back to coarse so the
            # rendered [Task list] doesn't keep claiming a close look that
            # isn't happening.
            fine_used, spent = prev_used, True
            if detail == "fine":
                detail = "coarse"
        elif incoming_detail == "fine" and prev_detail != "fine":
            fine_used, spent = 0, False   # genuine new close-look request
        else:
            fine_used, spent = prev_used, prev_spent

        normalized.append({
            "id": item_id,
            "content": content,
            "status": status,
            "note": item.get("note"),
            # Preserved across edits like id/content — the model rewrites
            # the item list on every update_task_list call but doesn't know
            # about (and shouldn't have to resend) prior observations.
            "observations": (existing_item or {}).get("observations", []),
            # The reasoning model's own brief to the vision stage for this
            # item — "read the ingredient list, flag any gelatin". Unlike
            # observations this IS the model's to set, so an incoming value
            # wins; absent one we preserve what was already there rather
            # than dropping the standing query on every unrelated full
            # replace. See agent.py's vision-prompt construction, which
            # prefers this over inferring intent from the item text.
            "watch_for": (item.get("watch_for") or "").strip()
            or (existing_item or {}).get("watch_for")
            or None,
            # How closely the camera has to look for this step. "fine" costs
            # roughly 2.5x the image tokens of "coarse" (full resolution vs the
            # 640px live-tick cap), so it is opt-in and scoped to one item —
            # when the step stops being in_progress the cost stops with it.
            # See agent.py's vision stage and app.js's capture sizing.
            "detail": detail,
            # Frames already spent looking closely at this item. Lives on the
            # item rather than in memory so it survives a Render restart —
            # the budget is meaningless if a restart silently refills it.
            "fine_frames_used": fine_used,
            # Set once the budget runs out, cleared only when the model
            # explicitly sends detail:"coarse". Without it the auto-revert
            # above is indistinguishable from the model choosing coarse, and
            # the next full replace refills the budget.
            "fine_budget_spent": spent,
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

    # Protect an active live-search goal from being silently wiped by a
    # full-replace that omitted it. request_live_search's start_find_task()
    # appends a "Find <target>" item (in_progress) as a side effect; when the
    # model calls update_task_list in the SAME turn, it wrote that call before
    # start_find_task had run, so its full item list can't yet include the find
    # item. Tool execution order within a turn isn't guaranteed, so without
    # this the just-registered goal — which the live-frame silence/observation
    # machinery checks every frame against — can disappear the instant it's
    # created. Only in_progress find-goals are protected: once the target's
    # been found (add_observation marks the item completed) the model is free
    # to drop or keep it like any other item.
    new_contents = {i["content"].lower() for i in normalized}
    for item in existing.get("items", []):
        if (
            item.get("status") == "in_progress"
            and is_find_goal_content(item.get("content", ""))
            and item.get("content", "").lower() not in new_contents
        ):
            document["items"].append(item)
            logger.info(
                f"set_document: preserved active find-goal {item['content']!r} "
                "that an incoming full-replace omitted."
            )

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

FIND_GOAL_PREFIX = "find "


def _wants_fine(item: dict) -> bool:
    """An in-progress item asking for a close look that still has budget."""
    return (
        item.get("status") == "in_progress"
        and item.get("detail") == "fine"
        and item.get("fine_frames_used", 0) < MAX_FINE_FRAMES_PER_ITEM
    )


def active_detail() -> str:
    """The detail level the camera should be capturing at right now.

    "fine" if any in-progress item asked for it *and* has fine-frame budget
    left, else "coarse". Read by the vision stage and echoed to the client,
    which sizes its next capture from it — the resolution decision has to
    reach the browser, because a frame downscaled before upload can never be
    inspected closely afterwards.

    Deliberately a pure read: it runs on every response to echo frame_detail,
    so the spending half lives in record_fine_frame() where it happens once
    per frame actually looked at.
    """
    document = get_document()
    if not document:
        return "coarse"
    return "fine" if any(_wants_fine(i) for i in document.get("items", [])) else "coarse"


def record_fine_frame() -> None:
    """Charge one fine frame against every in-progress item asking for one,
    and drop any that just ran out back to "coarse".

    Called from the vision stage after a frame has actually been looked at
    closely — not from active_detail(), which is a read consulted many times
    per turn and would over-charge wildly.

    Writing the exhausted item back to detail:"coarse" rather than only
    letting active_detail() refuse it is deliberate: `detail` is rendered into
    the prompt, so the model sees the close look has ended and can say so or
    re-request it explicitly, instead of silently getting coarse frames while
    the list still claims a close look is running.
    """
    document = get_document()
    if not document:
        return

    changed = False
    for item in document.get("items", []):
        if not _wants_fine(item):
            continue
        item["fine_frames_used"] = item.get("fine_frames_used", 0) + 1
        changed = True
        if item["fine_frames_used"] >= MAX_FINE_FRAMES_PER_ITEM:
            item["detail"] = "coarse"
            # Marks the revert as ours, not the model's — see set_document.
            item["fine_budget_spent"] = True
            logger.info(
                f"record_fine_frame: {item['content']!r} used its "
                f"{MAX_FINE_FRAMES_PER_ITEM}-frame close-look budget — "
                "reverted to coarse."
            )

    if changed:
        DOCUMENT_FILE.write_text(json.dumps(document, indent=2))


def is_find_goal_content(content: str) -> bool:
    return content.strip().lower().startswith(FIND_GOAL_PREFIX)


def is_find_goal(item_ref: str) -> bool:
    """True if `item_ref` names a live-search find-goal — a "Find X" item, the
    only kind that can be *completed* by spotting something.

    Read by agent.py to decide whether a log_observation(found=True) should
    also close the camera (goal_complete). Matching is on the item's text,
    which never changes, so this answers the same before and after
    add_observation flips the status.
    """
    document = get_document()
    if not document:
        return False
    ref = item_ref.strip().lower()
    return any(
        (i["id"] == item_ref or i["content"].lower() == ref)
        and is_find_goal_content(i["content"])
        for i in document.get("items", [])
    )


def add_observation(item_ref: str, note: str, found: bool = False) -> str:
    """Append a short note to whichever task-list item `item_ref` matches
    (by id or, case-insensitively, by content). Capped per item so the
    prompt injection in render_summary stays small regardless of session
    length — older notes are dropped, not the whole log.

    `found=True` marks the matched item "completed" — but ONLY for a
    "Find X" item (registered by request_live_search/start_find_task), which
    is by definition done once the target's been spotted.

    That guard is the fix for a real failure: `found` is documented to the
    model as "important enough to guarantee they're told", so it reasonably
    set it on ordinary progress notes. On a cooking STEP that silently
    completed the step — one real session marked "Prepare tadka" done off a
    note reading "oil poured into empty pan, no ingredients added yet", and
    (via agent.py's goal_complete) closed the camera too. A step is finished
    when the work is finished, never because a frame was worth mentioning;
    only "speak this out loud" is now carried by the separate `alert` flag.
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

    completed = False
    if found and is_find_goal_content(match["content"]):
        match["status"] = "completed"
        completed = True

    DOCUMENT_FILE.write_text(json.dumps(document, indent=2))
    if found and not completed:
        # Tell the model plainly rather than failing silently — it asked to
        # end something that isn't a search, and needs to know the step is
        # still open so it doesn't move on.
        logger.info(
            f"add_observation: ignored found=True on non-find item "
            f"{match['content']!r} — logged as a plain observation."
        )
        return (
            f"Logged observation for '{match['content']}'. Note: found=true only "
            "applies to a 'Find X' search goal, so this step was NOT marked "
            "completed and is still in progress. To say something out loud, use "
            "alert=true; to finish a step, use update_task_list."
        )
    return f"Logged observation for '{match['content']}'."


def retract_observation(item_ref: str, note_match: str, reopen: bool = False) -> str:
    """Delete observations on `item_ref` whose text contains `note_match`, and
    optionally move a wrongly-completed item back to in_progress.

    The counterpart to add_observation, which was append-only. From the
    2026-07-28 session: the model logged found=true for "toor dal" in a plastic
    bag, the user said *"the thing in the plastic bag is black eyed beans not
    toor dal"* — and nothing could act on that. The observation stayed, the
    item stayed completed, the wrong note kept riding along in every subsequent
    [Task list] injection, and the model confabulated from it for four turns,
    was challenged twice, apologised, and then repeated the claim.

    **A false observation that survives an explicit correction is worse than no
    observation at all**, because once it is in the log it is indistinguishable
    from a verified one.

    Substring matching rather than an index: the model is working from the
    rendered [Task list] text and can quote a fragment of the wrong note far
    more reliably than it can count list positions. Matching is case-insensitive
    and removes every observation that matches, since a wrong fact tends to have
    been logged more than once across ticks.

    `reopen` also clears the item's `note` when it matches, because that field
    is the other place a bad fact rides along, and un-completing an item while
    leaving the note that justified completing it just recreates the problem.
    """
    document = get_document()
    if not document or not document.get("items"):
        return "No active task list — nothing to retract."

    match = next(
        (i for i in document["items"]
         if i["id"] == item_ref or i["content"].lower() == item_ref.strip().lower()),
        None,
    )
    if not match:
        return f"No task list item matching '{item_ref}' — check the [Task list] content exactly."

    needle = note_match.strip().lower()
    if not needle:
        return "retract_observation needs the text of the observation to remove."

    before = match.get("observations", [])
    kept = [o for o in before if needle not in o.lower()]
    removed = len(before) - len(kept)
    match["observations"] = kept

    reopened = False
    if reopen and match["status"] in ("completed", "skipped"):
        match["status"] = "in_progress"
        reopened = True
    note_cleared = False
    if reopen and match.get("note") and needle in match["note"].lower():
        match["note"] = None
        note_cleared = True

    if not removed and not reopened and not note_cleared:
        return (
            f"Nothing matched '{note_match}' on '{match['content']}'. Its current "
            f"observations are: {kept or 'none'}. Quote a fragment of the exact "
            "wrong note, or pass reopen=true to re-open the item."
        )

    DOCUMENT_FILE.write_text(json.dumps(document, indent=2))
    logger.info(
        f"retract_observation: {match['content']!r} — removed {removed} "
        f"observation(s), reopened={reopened}, note_cleared={note_cleared}."
    )
    bits = []
    if removed:
        bits.append(f"removed {removed} observation(s)")
    if reopened:
        bits.append("re-opened the item (now in progress)")
    if note_cleared:
        bits.append("cleared its note")
    return (
        f"Corrected '{match['content']}': {', '.join(bits)}. Treat the retracted "
        "information as false from now on — do not repeat it."
    )


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
        # Only for the step actually underway: this is the standing brief the
        # vision stage is running right now, so it's shown so the model can
        # see what it previously asked for and revise it. On a pending or
        # finished item it isn't driving anything, and would just be prompt
        # bulk on every future turn.
        if item["status"] == "in_progress" and item.get("watch_for"):
            close = " (close-up)" if item.get("detail") == "fine" else ""
            lines.append(f"    watching for{close}: {item['watch_for']}")
        if not observations or (lean and item["status"] in ("completed", "skipped")):
            continue
        for obs in item.get("observations", []):
            lines.append(f"    - {obs}")
    return "\n".join(lines)
