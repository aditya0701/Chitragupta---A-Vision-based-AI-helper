"""Agentic core — tool registry, memory, and the main agent loop."""

from __future__ import annotations
import json
import re
from typing import Any, Callable, Optional


class Tool:
    """A tool the agent can invoke."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., str],
        parameters: dict[str, dict],
        needs_followup: bool = True,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters
        # Whether a call to this tool warrants a second Groq call to weave its
        # result into the reply. True for tools that surface new information
        # (web_search, fetch_page) the model hasn't seen yet. False for tools
        # that are pure side effects with a self-explanatory result
        # (start_timer, update_task_list) — the model's own visible text
        # around the tool call already says what it needs to; regenerating it
        # would just be a second paid call to restate the same thing.
        self.needs_followup = needs_followup

    def to_openai_tool(self) -> dict:
        # "required" is a per-parameter flag in our own Tool definitions
        # (convenient for the prompt-text renderer in agent.py), but real
        # JSON Schema wants it as a sibling list of names, not a property of
        # each property — leaving it inline was harmless noise for the
        # never-used old to_openai_tools() path, but native tool calling
        # (added 2026-07-13) actually sends this schema to the API, so it
        # needs to be valid.
        properties = {k: {kk: vv for kk, vv in v.items() if kk != "required"}
                      for k, v in self.parameters.items()}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [k for k, v in self.parameters.items() if v.get("required")],
                },
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]


# ─── Built-in tools ───────────────────────────────────────────────────────────

def tool_web_search(query: str) -> str:
    """Search the web via DuckDuckGo's HTML endpoint (no API key required)."""
    import httpx
    from bs4 import BeautifulSoup

    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; Chitragupt/1.0)"},
            timeout=10.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Web search failed: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result")[:5]:
        title_el = result.select_one(".result__title")
        snippet_el = result.select_one(".result__snippet")
        link_el = result.select_one(".result__url")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        url = link_el.get_text(strip=True) if link_el else ""
        results.append(f"- {title} ({url})\n  {snippet}")

    if not results:
        return f'No web search results found for "{query}".'
    return f'Web search results for "{query}":\n' + "\n".join(results)


def tool_fetch_page(url: str) -> str:
    """Fetch a web page and return its visible text content."""
    import httpx
    from bs4 import BeautifulSoup

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Chitragupt/1.0)"},
            timeout=10.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Failed to fetch page: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return text


def tool_calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval — only allow math
        import math
        allowed = {"abs": abs, "round": round, "int": int, "float": float, "min": min, "max": max, "sum": sum, "math": math}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_time(timezone: str = "UTC") -> str:
    """Get the current time (stub)."""
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc)
    return f"Current UTC time: {now.isoformat()}"


def tool_start_timer(label: str, duration_seconds: int, context: str = "") -> str:
    """Start a persisted background timer (no LLM cost while it runs)."""
    from . import timers
    timer_id = timers.start_timer(label, int(duration_seconds), context)
    minutes = int(duration_seconds) // 60
    seconds = int(duration_seconds) % 60
    duration_str = f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"
    return f"Timer '{label}' started for {duration_str} (id: {timer_id})."


def tool_update_task_list(title: str, items: list) -> str:
    """Replace the current task/recipe document (like Claude Code's TodoWrite)."""
    from . import tasklist
    document = tasklist.set_document(title, items)
    counts: dict[str, int] = {}
    for it in document["items"]:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items()) or "empty"
    return f"Task list '{title}' updated ({summary})."


def tool_log_observation(item: str, note: str, found: bool = False) -> str:
    """Silently record a short fact against a task-list item — the substitute
    for re-describing the whole scene every turn: write the fact once, read
    it back later instead of needing the original image again.

    `found` doesn't change what's stored — it's read back by agent.py to
    decide whether this observation needs a guaranteed spoken alert (see the
    found_alert check in _process_locked). Tool-calling models frequently
    return an empty `content` alongside a tool call in the same completion
    (the API accepts text+tool_calls together, but this model reliably
    produces neither), so a live tick where the model just called
    log_observation with nothing else previously went completely silent even
    when the note said the target was found. `found=True` routes that case
    through the existing tool-result follow-up call instead of relying on
    the model to also write visible text in the same turn."""
    from . import tasklist
    return tasklist.add_observation(item, note, found=found)


def tool_request_camera() -> str:
    """Marker tool — never executed for its return value. Its presence in a
    response is intercepted specially in agent.py to ask the client for a
    fresh frame, since the server has no way to reach into the browser's
    camera itself."""
    return "CAMERA_REQUESTED"


def tool_request_live_search(target: str) -> str:
    """Registers the find-goal (so the live-frame silence/observation
    machinery has something to check frames against) and returns a marker.
    Actually starting the camera stream + polling is intercepted specially
    in agent.py/the client, same pattern as request_camera — the server
    can't turn on the browser's camera itself.
    """
    from . import tasklist
    tasklist.start_find_task(target)
    return "LIVE_SEARCH_REQUESTED"


def build_default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="Search the web for information",
        fn=tool_web_search,
        parameters={"query": {"type": "string", "description": "Search query", "required": True}},
    ))
    registry.register(Tool(
        name="fetch_page",
        description="Fetch a web page by URL and return its visible text content (e.g. to read a recipe list from a search result)",
        fn=tool_fetch_page,
        parameters={"url": {"type": "string", "description": "The URL to fetch", "required": True}},
    ))
    registry.register(Tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        fn=tool_calculate,
        parameters={"expression": {"type": "string", "description": "Math expression", "required": True}},
    ))
    registry.register(Tool(
        name="get_time",
        description="Get the current time",
        fn=tool_get_time,
        parameters={"timezone": {"type": "string", "description": "Timezone (default UTC)", "required": False}},
    ))
    registry.register(Tool(
        name="start_timer",
        description=(
            "Start a background timer for a cooking step or wait period (e.g. 'boil eggs 10 min'). "
            "Runs entirely server-side at no cost. When it completes, a follow-up message with the "
            "next step is generated automatically and delivered to the client — no need to check on it yourself."
        ),
        fn=tool_start_timer,
        parameters={
            "label": {"type": "string", "description": "Short name for what's being timed, e.g. 'eggs' or 'cake'", "required": True},
            "duration_seconds": {"type": "integer", "description": "How many seconds to wait", "required": True},
            "context": {"type": "string", "description": "Relevant recipe/task context to reference when the timer completes", "required": False},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="update_task_list",
        description=(
            "Create or update the current task/recipe document — the persistent record of what "
            "needs doing, what's done, and what's been substituted. Always send the FULL list of "
            "items every time, not just the one that changed (like rewriting a todo list in full "
            "on each edit) — items you omit are dropped. Statuses: 'pending' (not started), "
            "'in_progress' (currently doing), 'completed' (done — keep it in the list, don't remove "
            "it), 'skipped' (substituted or skipped — put why in 'note'). Use this instead of "
            "repeating the whole plan back to the user in every reply."
        ),
        fn=tool_update_task_list,
        parameters={
            "title": {"type": "string", "description": "Name of the overall task, e.g. 'Chicken Biryani'", "required": True},
            "items": {
                "type": "array",
                "description": "Full list of task items, sent in full every time — anything omitted is dropped.",
                "required": True,
                # Structural schema, not just prose — this is what actually
                # constrains native tool calling to the right field names
                # (was previously only described in free text, which let
                # the model drift to writing "task" instead of "content").
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The item's text — always use this exact key, never 'task' or 'label'"},
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
        name="log_observation",
        description=(
            "Silently record a short factual note against a task-list item — "
            "e.g. what you just saw relevant to it. Does not produce a reply "
            "to the user by itself; call this on every frame that's relevant "
            "to an active item, whether or not you also decide to say "
            "something out loud this turn. Set found=true if this note means "
            "you found the thing the user is looking for (or another change "
            "important enough to guarantee they're told) — this triggers a "
            "spoken alert on its own, so the user is notified even if you "
            "don't separately write visible text this turn."
        ),
        fn=tool_log_observation,
        parameters={
            "item": {"type": "string", "description": "The exact task-list item content (or its id) this observation is about", "required": True},
            "note": {"type": "string", "description": "Short factual note, e.g. 'freezer drawer open, chicken tenders visible, no ice cream'", "required": True},
            "found": {"type": "boolean", "description": "True if this note means the target was found or something important changed — guarantees a spoken alert", "required": False},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="request_camera",
        description=(
            "Ask for a single fresh camera frame when answering the current "
            "message genuinely requires seeing the scene right now and no "
            "image is attached to this message. One frame, one look — use "
            "request_live_search instead if a single frame won't be enough. "
            "Only usable when no image was already provided this turn. Do "
            "not guess an answer that depends on the current scene without "
            "calling this first. This is one of several tools available to "
            "you, not a headline feature — don't volunteer it unprompted."
        ),
        fn=tool_request_camera,
        parameters={},
        needs_followup=False,
    ))
    registry.register(Tool(
        name="request_live_search",
        description=(
            "Start continuously watching the camera when the user needs "
            "help locating a specific physical object and a single frame "
            "won't be enough — they'll need to move the camera around "
            "while you keep checking. This ONLY watches for the named "
            "target; do not use it for general cooking guidance, task "
            "tracking, or any other kind of ongoing help — that's not "
            "enabled through this tool. Once started, stay silent on "
            "frames that don't show the target (the live-frame protocol "
            "handles this automatically) and speak up only when you "
            "actually see it, or if the user seems to be searching the "
            "wrong place. This is one of several tools available to you, "
            "not a headline feature — don't volunteer it unprompted."
        ),
        fn=tool_request_live_search,
        parameters={
            "target": {"type": "string", "description": "The specific thing to look for, in the user's own words", "required": True},
        },
        needs_followup=False,
    ))
    return registry


# ─── Conversation Memory ──────────────────────────────────────────────────────

class ConversationMemory:
    """Simple in-memory conversation history."""

    def __init__(self, max_turns: int = 50):
        self.history: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self) -> list[dict]:
        return self.history

    def clear(self):
        self.history = []
