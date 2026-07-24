"""Tool registry for the live system.

Reuses the Tool/ToolRegistry classes and the info-gathering tool functions
(web_search, fetch_page, calculate) from server.agent unchanged. The
doc-mutating tools are new: they close over a `get_doc` callable supplied by
LiveAgent, mutating the in-memory doc for the current turn — the agent owns
loading it at turn start and persisting it at turn end, so a turn's tool
calls and the agent's own writes can never interleave on disk.

Deliberately absent vs. the old system:
  start_timer        subsumed by a time-anchored expectation — same wall-
                     clock arithmetic, plus a resolution path timers never
                     had (satisfied silently if confirmed before deadline).
  request_camera /   the live UI owns the camera; chat turns attach the
  request_live_search  current frame client-side whenever the stream is on.
"""

from __future__ import annotations

from typing import Callable

from ..agent import (
    Tool,
    ToolRegistry,
    tool_calculate,
    tool_fetch_page,
    tool_web_search,
)
from . import worlddoc


def build_live_tools(get_doc: Callable[[], dict]) -> ToolRegistry:
    registry = ToolRegistry()

    def _update_tasks(title: str, items: list) -> str:
        return worlddoc.set_tasks(get_doc(), title, items)

    def _set_expectation(
        description: str,
        anchor: str,
        due_in_seconds: float = None,
        condition: str = None,
        priority: str = "normal",
        task: str = None,
    ) -> str:
        return worlddoc.add_expectation(
            get_doc(), description, anchor,
            due_in_seconds=due_in_seconds, condition=condition,
            priority=priority, task_ref=task,
        )

    def _resolve_expectation(ref: str, outcome: str = "satisfied", note: str = "") -> str:
        return worlddoc.resolve_expectation(get_doc(), ref, outcome, note)

    def _log_environment(fact: str) -> str:
        return worlddoc.add_environment_fact(get_doc(), fact)

    def _mark_task(task: str, status: str, note: str = "") -> str:
        doc = get_doc()
        match = worlddoc.find_task(doc, task)
        if not match:
            return f"No task matching '{task}' — check the [Tasks] content exactly."
        if status not in worlddoc.VALID_TASK_STATUSES:
            return f"status must be one of {sorted(worlddoc.VALID_TASK_STATUSES)}."
        match["status"] = status
        if note:
            match["note"] = note
        worlddoc.touch_task(doc, match["id"])
        return f"Task '{match['content']}' marked {status}."

    registry.register(Tool(
        name="update_tasks",
        description=(
            "Create or fully replace the plan — the persistent record of what needs doing. "
            "Send the FULL list every time (items you omit are dropped). Statuses: pending, "
            "in_progress, completed (keep in list), skipped (say why in note). For changing "
            "ONE item's status, prefer mark_task instead of resending everything."
        ),
        fn=_update_tasks,
        parameters={
            "title": {"type": "string", "description": "Name of the overall goal, e.g. 'Chicken Biryani'", "required": True},
            "items": {
                "type": "array",
                "description": "Full list of task items.",
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The item's text — always this exact key, never 'task' or 'label'"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "skipped"]},
                        "note": {"type": "string", "description": "Optional, e.g. reason for a substitution"},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="mark_task",
        description="Change one task's status without resending the whole list.",
        fn=_mark_task,
        parameters={
            "task": {"type": "string", "description": "Exact task content or id", "required": True},
            "status": {"type": "string", "description": "pending | in_progress | completed | skipped", "required": True},
            "note": {"type": "string", "description": "Optional note", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="set_expectation",
        description=(
            "Register something that SHOULD happen so the system can notice if it doesn't. "
            "Two kinds: anchor='time' with due_in_seconds (e.g. 'rice should be started', due "
            "in 600s — if not confirmed by then, the user gets reminded automatically, at zero "
            "cost while waiting). anchor='event' with condition (e.g. 'verify all items "
            "collected', condition: 'user appears to be at the checkout / heading to the door' "
            "— you'll check the condition yourself against future frames). Set priority='high' "
            "only for things worth interrupting the user over; 'low' waits for a natural moment. "
            "When you help plan anything with timings, set the expectations in the same turn — "
            "don't wait to be asked."
        ),
        fn=_set_expectation,
        parameters={
            "description": {"type": "string", "description": "What should happen, phrased so it's useful when read back later", "required": True},
            "anchor": {"type": "string", "enum": ["time", "event"], "required": True},
            "due_in_seconds": {"type": "number", "description": "time-anchored only: seconds from now until this is overdue", "required": False},
            "condition": {"type": "string", "description": "event-anchored only: what visible situation makes this fire", "required": False},
            "priority": {"type": "string", "enum": ["high", "normal", "low"], "required": False},
            "task": {"type": "string", "description": "Optional: task content/id this belongs to", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="resolve_expectation",
        description=(
            "Close an open expectation: outcome='satisfied' when you've confirmed it happened "
            "(from the frame or the user saying so), 'cancelled' when it no longer applies. "
            "Resolve satisfied expectations silently as you notice them — no need to announce it."
        ),
        fn=_resolve_expectation,
        parameters={
            "ref": {"type": "string", "description": "Expectation id or exact description", "required": True},
            "outcome": {"type": "string", "enum": ["satisfied", "cancelled"], "required": False},
            "note": {"type": "string", "description": "Optional context", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="log_environment",
        description=(
            "Record a durable fact about the physical environment worth remembering beyond this "
            "moment — where things are, what's available, what state something was left in. "
            "E.g. 'red chili powder is on the top shelf, left side'. Use the same place-words "
            "consistently across the session ('top shelf' stays 'top shelf') so facts stay "
            "matchable. This is your long-term spatial memory — log locations whenever you spot "
            "something the user might look for later, even if it's not relevant right now."
        ),
        fn=_log_environment,
        parameters={
            "fact": {"type": "string", "description": "One short, specific, durable fact", "required": True},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="web_search",
        description="Search the web for information (e.g. a recipe's real timings before planning).",
        fn=tool_web_search,
        parameters={"query": {"type": "string", "description": "Search query", "required": True}},
    ))
    registry.register(Tool(
        name="fetch_page",
        description="Fetch a web page by URL and return its visible text content.",
        fn=tool_fetch_page,
        parameters={"url": {"type": "string", "description": "The URL to fetch", "required": True}},
    ))
    registry.register(Tool(
        name="calculate",
        description="Evaluate a mathematical expression.",
        fn=tool_calculate,
        parameters={"expression": {"type": "string", "description": "Math expression", "required": True}},
    ))

    return registry
