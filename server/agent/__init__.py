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
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters

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
    """Search the web (stub — integrate with a real search API)."""
    return f'[Web search results for "{query}"]'


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


def build_default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="Search the web for information",
        fn=tool_web_search,
        parameters={"query": {"type": "string", "description": "Search query", "required": True}},
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
