"""The main Chitragupt agent — orchestrates vision + tools + memory.

On the Groq API backend, one multimodal model handles both vision and
reasoning in a single call. The two-stage vision/reasoning split (separate
Ollama models) only applies to backends with SPLIT_VISION_REASONING set,
e.g. Colab.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Optional

from ..backends import VisionBackend, VisionResponse, should_think
from ..config import settings
from . import ToolRegistry, ConversationMemory, build_default_tools, timers, tasklist

logger = logging.getLogger("chitragupt")


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
    2. For backends with separate vision/reasoning models (SPLIT_VISION_REASONING):
       calls backend.vision() first (Stage 1), then backend.chat() with the
       resulting description (Stage 2) — two calls, only when unavoidable.
       For single multimodal backends: passes the image straight into
       backend.chat() alongside the prompt — one call.
    3. Parses tool calls from the response
    4. Executes tools and returns results
    5. Maintains conversation memory
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

        `is_live_frame` marks an automated live-streaming ping rather than a
        real user question — its prompt is not recorded in conversation
        memory, so routine "watching" frames don't crowd out real turns.
        """
        if not is_live_frame:
            self.memory.add("user", prompt)

        split_stages = image_base64 and self.backend.SPLIT_VISION_REASONING

        # ── Stage 1: Vision (Colab only) ────────────────────────────────
        # Only Colab's split qwen3-vl + qwen3 setup sets SPLIT_VISION_REASONING,
        # so this whole block is skipped under the current Groq/API setup —
        # kept here for when Colab is used again. API-mode backends pass the
        # image straight into the single reasoning call below instead.
        scene_description = None
        if split_stages:
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
        # Decide once, from the raw user prompt — not the wrapped reasoning
        # prompt below, which always exceeds any length heuristic once the
        # system framing and scene context are added.
        think = should_think(prompt)

        # Build the reasoning prompt with scene context
        reason_prompt = self._build_reason_prompt(
            prompt=prompt,
            scene=scene_description,
            has_image=bool(image_base64) and not split_stages,
            think=think,
        )

        response = await self.backend.chat(
            # Split-stage backends already consumed the image in Stage 1
            # above; single-call backends get it here alongside the prompt.
            image_base64=None if split_stages else image_base64,
            prompt=reason_prompt,
            conversation_history=self.memory.get_history()[-10:],
            think=think,
        )

        full_text = response.text

        if response.reasoning:
            # Backend already separated reasoning from the answer (e.g. Groq's
            # reasoning_format="parsed") — nothing to strip, trust it as-is.
            think_blocks = [response.reasoning]
            clean_text = full_text.strip()
        else:
            # Inline-tag convention (local Ollama/Qwen3): reasoning is mixed
            # into the same text wrapped in <think>...</think>.
            think_blocks = self._extract_think_blocks(full_text)
            clean_text = self._remove_think_blocks(full_text).strip()

        tool_results = []
        if settings.TOOLS_ENABLED:
            # Only scan the *visible* response for tool calls, not the raw
            # thinking trace — the model often mentions tool syntax
            # hypothetically while reasoning about whether to use one, and
            # scanning full_text (thinking included) treated that hypothetical
            # mention as a real invocation, triggering a wasted second API call.
            tool_results = await self._execute_tool_calls(clean_text)
            # Unresolved matches (unknown tool name, malformed JSON) aren't
            # worth a costly follow-up call — only resolved tool calls should
            # trigger one.
            tool_results = [
                r for r in tool_results
                if not r["result"].startswith("Unknown tool:")
                and not r["result"].startswith("JSON parse error:")
            ]

        if tool_results and any(self.tools.get(r["tool"]).needs_followup for r in tool_results):
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
        elif tool_results:
            # Every tool called was a pure side effect (e.g. start_timer,
            # update_task_list) — the model's own surrounding text already
            # says what happened, so skip the paid follow-up call and just
            # strip the raw tool-call syntax out of what's shown to the user.
            # If the model wrote nothing but the tool call itself, fall back
            # to the tool's own confirmation string rather than showing
            # nothing (or, worse, the raw unstripped JSON).
            final_text = self._strip_tool_blocks(clean_text) or "\n".join(
                r["result"] for r in tool_results
            )
        else:
            final_text = clean_text or full_text

        # Fold in any timer that finished while we were talking, so a
        # completion surfaces immediately in this reply instead of waiting
        # for the next background /v1/timers/check poll tick.
        timer_update = await self.check_timers()
        if timer_update["completed"]:
            timer_lines = "\n".join(
                f"⏰ {t['label']}: {t['message']}" for t in timer_update["completed"]
            )
            final_text = f"{final_text}\n\n{timer_lines}" if final_text else timer_lines

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

    def _build_reason_prompt(
        self,
        prompt: str,
        scene: Optional[str],
        has_image: bool = False,
        think: bool = True,
    ) -> str:
        """Build the prompt for the reasoning model."""
        parts = [
            "You are Chitragupt, an all-seeing assistant."
            if not settings.TOOLS_ENABLED
            else "You are Chitragupt, an all-seeing assistant with access to tools.",
        ]

        if scene:
            parts.append(f"\n[Camera feed]\n{scene}")
        elif has_image:
            parts.append(
                "\n[Camera feed attached]\nAn image is attached below — look at "
                "it directly to answer, describing relevant details as needed."
            )

        doc_summary = tasklist.render_summary(tasklist.get_document())
        if doc_summary:
            parts.append(f"\n[Task list]\n{doc_summary}")

        parts.append(f"\n[User]\n{prompt}")

        tool_instruction = ""
        if settings.TOOLS_ENABLED:
            tool_list = "\n".join(
                f"- {t.name}({', '.join(t.parameters)}): {t.description}"
                for t in self.tools.list_tools()
            )
            tool_instruction = (
                "\n\nYou have tools available. To call one, write this in your "
                "visible response, not inside a <think> block (only the visible "
                "response is checked for tool calls):\n"
                '```tool\n{"name": "tool_name", "arguments": {"arg1": "value"}}\n```\n'
                f"Available tools:\n{tool_list}\n\n"
                "Tool-specific guidance:\n"
                "- start_timer: use for any step that needs waiting (boiling, baking, "
                "marinating, steeping). It runs in the background for free — don't wait "
                "for it or ask about it again yourself; completion is announced to the "
                "user automatically when it's done. Start it, then keep helping with "
                "whatever's next.\n"
                "- update_task_list: use whenever you're guiding a multi-step task (a "
                "recipe, a shopping list, a project). Always send the FULL item list, "
                "even items already completed — anything you leave out is dropped. Mark "
                "finished items 'completed' rather than removing them, and use 'skipped' "
                "with a note for substitutions. If a [Task list] is shown above, read it "
                "before updating it, and don't just recite the whole plan back in your "
                "reply — the user can already see it."
            )
        thinking_instruction = (
            "\n\nThink step by step before responding."
            if think
            else "\n\nAnswer directly and concisely — no need for extended reasoning."
        )
        parts.append(
            thinking_instruction
            + tool_instruction
            + " Be concise, practical, and helpful in your final response."
            + " Respond in plain text only — no markdown (no **bold**, no headers, "
            "no bullet/numbered lists with * or -). This response may be read aloud "
            "by text-to-speech, so write it as plain spoken sentences."
        )

        return "\n".join(parts)

    def _extract_think_blocks(self, text: str) -> list[str]:
        """Extract <think>...</think> blocks from Qwen3 output."""
        return re.findall(r"<think>(.*?)</think>", text, re.DOTALL)

    def _remove_think_blocks(self, text: str) -> str:
        """Strip <think>...</think> blocks to get the visible response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _strip_tool_blocks(self, text: str) -> str:
        """Remove raw tool-call syntax, leaving only the model's own prose."""
        text = re.sub(r"```tool\s*\n?{.*?}\n?```", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

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
        """Clear conversation memory, frame buffer, and any active task document."""
        self.memory.clear()
        self.frame_buffer.clear()
        tasklist.clear_document()

    async def check_timers(self) -> dict:
        """Fire completion calls for any due timers, return completions + free progress.

        Called on every client poll, and opportunistically from process() too
        so an active conversation surfaces a completion immediately instead of
        waiting on the next poll tick. Checking due-ness and computing
        progress is pure arithmetic (timers.due_unfired / timers.active_progress)
        — Groq is only ever called once per timer, right here, only when it's
        actually done. Routed through the same prompt-building and
        tool-execution path as a normal turn, so the completion can also
        update the task list (mark the step done) rather than just narrate it.
        """
        for t in timers.due_unfired():
            timer_prompt = (
                f"[Timer completed]\nThe timer '{t['label']}' just finished after "
                f"{t['duration_seconds']} seconds.\nContext: {t['context'] or 'N/A'}\n"
                "Give a brief, practical next-step update for the user. If this step "
                "is tracked in the task list, update it to reflect that it's done."
            )
            reason_prompt = self._build_reason_prompt(
                prompt=timer_prompt, scene=None, has_image=False, think=False,
            )
            try:
                response = await self.backend.chat(image_base64=None, prompt=reason_prompt, think=False)
                clean_text = self._remove_think_blocks(response.text).strip()

                tool_results = []
                if settings.TOOLS_ENABLED:
                    tool_results = await self._execute_tool_calls(clean_text)
                    tool_results = [
                        r for r in tool_results
                        if not r["result"].startswith("Unknown tool:")
                        and not r["result"].startswith("JSON parse error:")
                    ]

                if tool_results and any(self.tools.get(r["tool"]).needs_followup for r in tool_results):
                    tool_context = "\n\n".join(
                        f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
                    )
                    final_response = await self.backend.chat(
                        image_base64=None,
                        prompt=(
                            f"I called tools while handling this timer completion. Results:\n\n"
                            f"{tool_context}\n\n{timer_prompt}\n"
                            "Please give the final brief update for the user."
                        ),
                    )
                    message = final_response.text.strip()
                elif tool_results:
                    message = self._strip_tool_blocks(clean_text) or "\n".join(
                        r["result"] for r in tool_results
                    )
                else:
                    message = clean_text

                timers.mark_fired(t["id"], message)
            except Exception as e:
                # Leave it unfired — it'll be retried on the next poll instead
                # of blocking other timers' progress from being returned this tick.
                logger.error(f"Timer completion call failed for '{t['label']}': {e}")

        return {
            "completed": timers.pop_completions(),
            "active": timers.active_progress(),
        }
