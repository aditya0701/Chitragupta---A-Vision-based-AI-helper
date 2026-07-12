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
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
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
            "items": {"type": "array", "description": "Full list of [{content, status, note?}, ...]", "required": True},
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
