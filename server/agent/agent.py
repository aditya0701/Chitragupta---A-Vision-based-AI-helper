"""The main Chitragupt agent — orchestrates vision + tools + memory.

Implements the two-stage pipeline from CLAUDE.md:
  Stage 1 (vision):  qwen3-vl:8b  →  text description
  Stage 2 (reason):  qwen3:8b     →  ReAct reasoning + tool calls + response

A memory buffer tracks the last N frame descriptions for change detection.
Note: `backend.vision()` (Stage 1) still runs on every frame passed in —
this buffer only skips Stage 2 (reasoning) when nothing has changed. Callers
that want to avoid the Stage 1 cost too (e.g. a live camera feed) need to
gate frames *before* calling `process()`, since the cost has already been
paid by the time this buffer sees the description.
"""

from __future__ import annotations
import json
import re
from typing import Optional

from ..backends import VisionBackend, VisionResponse
from . import ToolRegistry, ConversationMemory, build_default_tools


class FrameBuffer:
    """Rolling buffer of frame descriptions for change detection."""

    def __init__(self, max_frames: int = 10):
        self.frames: list[str] = []
        self.max_frames = max_frames

    def add(self, description: str):
        self.frames.append(description)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    @property
    def last(self) -> Optional[str]:
        return self.frames[-1] if self.frames else None

    @property
    def context(self) -> str:
        """Collapse buffered frames into a single context string."""
        if not self.frames:
            return ""
        return "\n---\n".join(
            f"[Frame {i+1}] {d}" for i, d in enumerate(self.frames)
        )

    def has_changed(self, new_description: str, min_words: int = 10) -> bool:
        """Simple change detection: compare word overlap with last frame."""
        if not self.frames:
            return True

        last_words = set(self.frames[-1].lower().split())
        new_words = set(new_description.lower().split())

        if len(last_words) < min_words or len(new_words) < min_words:
            return True

        jaccard = len(last_words & new_words) / len(last_words | new_words)
        return jaccard < 0.6  # less than 60% overlap = scene changed

    def clear(self):
        self.frames.clear()


class ChitraguptAgent:
    """
    The core agent that:
    1. Takes an image + prompt
    2. Calls backend.vision() to describe the image (Stage 1)
    3. Stores description in frame buffer for change detection
    4. Calls backend.chat() with context + prompt (Stage 2)
    5. Parses tool calls from the response
    6. Executes tools and returns results
    7. Maintains conversation memory
    """

    def __init__(self, backend: VisionBackend, tools: Optional[ToolRegistry] = None):
        self.backend = backend
        self.tools = tools or build_default_tools()
        self.memory = ConversationMemory()
        self.frame_buffer = FrameBuffer()

    async def process(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_live_frame: bool = False,
    ) -> dict:
        """Process a user request with optional image.

        Two-stage pipeline:
          1. Vision — describe the image (if provided)
          2. Reason — think + respond using the description

        `is_live_frame` marks an automated live-streaming ping rather than a
        real user question — its prompt is not recorded in conversation
        memory, so routine "watching" frames don't crowd out real turns.
        """
        if not is_live_frame:
            self.memory.add("user", prompt)

        # ── Stage 1: Vision ──────────────────────────────────────────────
        scene_description = None
        if image_base64:
            scene_description = await self.backend.vision(
                image_base64=image_base64,
                prompt=(
                    "Describe everything visible in this image in detail. "
                    "Include: objects, people, actions, text, colours, spatial layout, "
                    "and anything that might matter for helping someone understand this scene."
                ),
            )

            # Check if scene has meaningfully changed
            if not self.frame_buffer.has_changed(scene_description):
                self.frame_buffer.add(scene_description)
                return {
                    "text": "👁️ Scene unchanged — still monitoring.",
                    "model": "qwen3-vl:8b",
                    "provider": "colab",
                    "tool_calls": [],
                    "scene_unchanged": True,
                    "scene_description": scene_description,
                }

            self.frame_buffer.add(scene_description)

        # ── Stage 2: Reason ──────────────────────────────────────────────
        # Build the reasoning prompt with scene context
        reason_prompt = self._build_reason_prompt(
            prompt=prompt,
            scene=scene_description,
        )

        response = await self.backend.chat(
            image_base64=None,  # image already processed in stage 1
            prompt=reason_prompt,
            conversation_history=self.memory.get_history()[-10:],
        )

        full_text = response.text

        # Extract think blocks for logging
        think_blocks = self._extract_think_blocks(full_text)
        clean_text = self._remove_think_blocks(full_text).strip()

        # Check for tool calls
        tool_results = await self._execute_tool_calls(full_text)

        if tool_results:
            tool_context = "\n\n".join(
                f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
            )
            final_prompt = (
                f"I called tools to answer the user. Here are the results:\n\n"
                f"{tool_context}\n\n"
                f"Original question: {prompt}\n"
                f"Scene context: {scene_description or 'N/A'}\n"
                f"Please provide a final answer incorporating these results."
            )
            final_response = await self.backend.chat(
                image_base64=None,
                prompt=final_prompt,
            )
            final_text = final_response.text
        else:
            final_text = clean_text or full_text

        if not is_live_frame:
            self.memory.add("assistant", final_text)

        return {
            "text": final_text,
            "model": response.model,
            "provider": response.provider,
            "tool_calls": tool_results or [],
            "think_blocks": think_blocks,
            "scene_description": scene_description,
        }

    def _build_reason_prompt(self, prompt: str, scene: Optional[str]) -> str:
        """Build the prompt for the reasoning model."""
        parts = [
            "You are Chitragupt, an all-seeing assistant with access to tools.",
        ]

        if scene:
            parts.append(f"\n[Camera feed]\n{scene}")

        parts.append(f"\n[User]\n{prompt}")

        parts.append(
            "\n\nThink step by step before responding. "
            "If you need external information, call a tool inside your thinking. "
            "Be concise, practical, and helpful in your final response."
        )

        return "\n".join(parts)

    def _extract_think_blocks(self, text: str) -> list[str]:
        """Extract <think>...</think> blocks from Qwen3 output."""
        return re.findall(r"<think>(.*?)</think>", text, re.DOTALL)

    def _remove_think_blocks(self, text: str) -> str:
        """Strip <think>...</think> blocks to get the visible response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    async def _execute_tool_calls(self, text: str) -> list[dict]:
        """Find and execute any tool calls in the response text.

        Supports two formats:
          - ```tool { "name": "...", "arguments": {...} } ```
          - <tool>tool_name: arg</tool>
        """
        results = []

        # Format 1: JSON tool blocks ```tool { ... } ```
        json_pattern = r"```tool\s*\n?({.*?})\n?```"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                call = json.loads(match.strip())
                tool_name = call.get("name")
                arguments = call.get("arguments", {})

                tool = self.tools.get(tool_name)
                if tool:
                    result = tool.fn(**arguments)
                    results.append({"tool": tool_name, "arguments": arguments, "result": result})
                else:
                    results.append({"tool": tool_name, "arguments": arguments, "result": f"Unknown tool: {tool_name}"})
            except json.JSONDecodeError as e:
                results.append({"tool": "unknown", "arguments": {}, "result": f"JSON parse error: {e}"})

        # Format 2: <tool>name: arg</tool> (Qwen3 ReAct format from CLAUDE.md)
        simple_pattern = r"<tool>(.*?):\s*(.*?)</tool>"
        matches = re.findall(simple_pattern, text, re.DOTALL)

        for tool_name, arg in matches:
            tool_name = tool_name.strip()
            arg = arg.strip()
            tool = self.tools.get(tool_name)
            if tool:
                result = tool.fn(arg)
                results.append({"tool": tool_name, "arguments": {"_": arg}, "result": result})
            else:
                results.append({"tool": tool_name, "arguments": {}, "result": f"Unknown tool: {tool_name}"})

        return results

    def reset_conversation(self):
        """Clear conversation memory and frame buffer."""
        self.memory.clear()
        self.frame_buffer.clear()
