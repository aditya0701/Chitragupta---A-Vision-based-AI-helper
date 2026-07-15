"""The main Chitragupt agent — orchestrates vision + tools + memory.

On the Groq API backend, one multimodal model handles both vision and
reasoning in a single call. The two-stage vision/reasoning split (separate
Ollama models) only applies to backends with SPLIT_VISION_REASONING set,
e.g. Colab.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import AsyncIterator, Optional

from ..backends import VisionBackend, VisionResponse, should_think
from ..config import settings
from . import ToolRegistry, ConversationMemory, build_default_tools, timers, tasklist

logger = logging.getLogger("chitragupt")

# Sentinel a live-frame turn writes as its entire visible reply to mean
# "nothing new relevant to the active goal" — stripped before display so it
# never leaks to the user, and never honored on a direct user turn (see
# _process_locked).
SILENT_MARKER = "[SILENT]"


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
        # Serializes every turn (typed chat, live-frame ping, timer
        # completion) so two of them can never interleave reads/writes of
        # the shared task-list document or timer state. Not reentrant —
        # internal callers must use the *_locked variants below, never
        # re-acquire this from within a turn that already holds it.
        self._lock = asyncio.Lock()

    async def process(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_live_frame: bool = False,
        is_camera_followup: bool = False,
    ) -> dict:
        async with self._lock:
            return await self._process_locked(image_base64, prompt, is_live_frame, is_camera_followup)

    async def _process_locked(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_live_frame: bool = False,
        is_camera_followup: bool = False,
    ) -> dict:
        """Process a user request with optional image.

        `is_live_frame` marks an automated live-streaming ping rather than a
        real user question — its prompt is not recorded in conversation
        memory, so routine "watching" frames don't crowd out real turns.

        `is_camera_followup` marks Phase B of a request_camera round trip —
        the client resending the *same* prompt text now that it has an
        image attached. The original prompt was already recorded when Phase
        A ran (see the request_camera short-circuit below), so recording it
        again here would duplicate the user's utterance in memory.
        """
        if not is_live_frame and not is_camera_followup:
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
        #
        # Previously this forced think=False for every imageless turn where
        # request_camera was on the table, on the theory that "no image yet"
        # meant "nothing to reason about." That conflated two different
        # things: a live tick's trivial "should I look?" check, and a
        # substantive imageless question like "help me plan chicken
        # biryani" — the one turn where getting the recipe/step breakdown
        # right actually matters most. Forcing shallow reasoning on that
        # turn was backwards. should_think(prompt) alone decides now; the
        # truncation-retry below is the actual safety net for a turn that
        # runs long, applied only when it actually happens rather than
        # preemptively on every imageless question.
        think = should_think(prompt)
        has_image = bool(image_base64) and not split_stages

        # Build the reasoning prompt with scene context
        reason_prompt = self._build_reason_prompt(
            prompt=prompt,
            scene=scene_description,
            has_image=has_image,
            think=think,
            is_live_frame=is_live_frame,
        )

        native_tools = (
            [t.to_openai_tool() for t in self._available_tools(has_image, is_live_frame)]
            if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
            else None
        )

        # Live ticks aren't recorded to memory and gain nothing from the
        # last 10 chat turns — the [Task list] block already injected into
        # reason_prompt is the durable state that matters here. Sending
        # full history on every tick was dead weight riding along on
        # exactly the calls most likely to hit the Groq TPM cap.
        history = None if is_live_frame else self.memory.get_history()[-10:]

        try:
            response = await self.backend.chat(
                # Split-stage backends already consumed the image in Stage 1
                # above; single-call backends get it here alongside the prompt.
                image_base64=None if split_stages else image_base64,
                prompt=reason_prompt,
                conversation_history=history,
                think=think,
                tools=native_tools,
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            # 413 ("this one request is too big") and 429 ("you've already
            # spent your rolling per-minute budget, this one just tipped it
            # over") are different failure shapes even though both come from
            # the same Groq TPM cap — a 413 shrinks with a leaner prompt, a
            # 429 doesn't, since the request itself may be perfectly small
            # and just arrived too soon after previous ones (e.g. rapid live
            # ticks while actively searching). Neither should surface the
            # raw provider error to the user.
            if status == 413:
                # Degrade once: drop every observation from the task-list
                # injection (not just completed/skipped ones) and force
                # think=False, then retry. If it fails again, let it raise —
                # one degrade attempt is enough to catch a near-miss, not a
                # systemic sizing problem.
                logger.warning(f"Backend rejected request as too large (413) — retrying with a stripped prompt: {e}")
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=scene_description, has_image=has_image,
                    think=think, is_live_frame=is_live_frame, strip_task_list=True,
                )
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=None,
                    think=think,
                    tools=native_tools,
                )
            elif status == 429:
                retry_after = self._parse_retry_after(e)
                if is_live_frame:
                    # Don't hold the shared lock waiting out a live tick's
                    # rate limit — another tick comes along in a few seconds
                    # anyway. Surface the wait so the frontend can back its
                    # own polling interval off instead of hammering the same
                    # limit again next tick.
                    logger.warning(f"Rate limited (429) on live tick — skipping this tick, suggested wait {retry_after}s.")
                    return {
                        "text": "",
                        "model": "n/a",
                        "provider": "n/a",
                        "tool_calls": [],
                        "think_blocks": [],
                        "scene_description": None,
                        "rate_limited": True,
                        "retry_after": retry_after,
                    }
                # A direct question deserves an actual answer — wait out the
                # provider's suggested delay once (capped, in case the
                # provider ever reports something unreasonable), then retry,
                # rather than surfacing a raw rate-limit error for something
                # the user is actively waiting on.
                logger.warning(f"Rate limited (429) — waiting {retry_after}s then retrying once.")
                await asyncio.sleep(min(retry_after, 10.0))
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=history,
                    think=think,
                    tools=native_tools,
                )
            else:
                raise

        if response.truncated:
            # Generation was cut off by max_tokens before the model finished.
            # Two genuinely different situations need different recoveries:
            if response.reasoning and not response.text.strip():
                # Reasoning finished cleanly (it's sitting in the thinking
                # box), but the model ran out of budget before ever writing
                # the actual answer. Don't throw that reasoning away and
                # re-derive it from scratch — feed it back and ask
                # specifically for the conclusion. Cheaper (short answer
                # only, no re-reasoning) and the answer is grounded in work
                # it already did instead of a fresh low-effort guess.
                logger.warning(
                    "Truncated with reasoning but no answer text — asking "
                    "the model to conclude from its own reasoning."
                )
                conclude_prompt = (
                    "You were reasoning through this and ran out of space "
                    "before writing your answer. Here is your own reasoning "
                    f"so far:\n\n{response.reasoning}\n\nBased on that, give "
                    "your final answer now — concise, no further reasoning "
                    f"needed.\n\nOriginal question: {prompt}"
                )
                response = await self.backend.chat(
                    image_base64=None, prompt=conclude_prompt, think=False, tools=native_tools,
                )
            else:
                # Reasoning itself got cut off or never separated cleanly —
                # nothing usable to hand back, so start over with a lower
                # reasoning budget rather than risk the same cutoff twice.
                logger.warning(
                    "Truncated with no usable separated reasoning — "
                    "retrying once from scratch with think=False."
                )
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=scene_description, has_image=has_image,
                    think=think, is_live_frame=is_live_frame,
                )
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=history,
                    think=think,
                    tools=native_tools,
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
            if response.tool_calls:
                # Native function-calling — structured, no parsing of the
                # visible text needed at all.
                tool_results = self._run_structured_tool_calls(response.tool_calls)
            else:
                # Fallback for backends without SUPPORTS_NATIVE_TOOLS. Only
                # scan the *visible* response for tool calls, not the raw
                # thinking trace — the model often mentions tool syntax
                # hypothetically while reasoning about whether to use one,
                # and scanning full_text (thinking included) treated that
                # hypothetical mention as a real invocation, triggering a
                # wasted second API call.
                tool_results = await self._execute_tool_calls(clean_text)
            # Unresolved matches (unknown tool name, malformed JSON) aren't
            # worth a costly follow-up call — only resolved tool calls should
            # trigger one.
            tool_results = [
                r for r in tool_results
                if not r["result"].startswith("Unknown tool:")
                and not r["result"].startswith("JSON parse error:")
                and not r["result"].startswith("Invalid arguments")
            ]

        # request_camera can't be resolved server-side — the image lives in
        # the browser. Short-circuit here instead of running the normal
        # tool-result/follow-up flow: tell the client to capture a frame and
        # resend this same question, rather than letting the model guess.
        camera_request = next((r for r in tool_results if r["tool"] == "request_camera"), None)
        if camera_request:
            final_text = self._strip_tool_blocks(clean_text) or "Let me take a look."
            # Not recorded to memory here — this is a provisional holding
            # message, not the real answer. The user's question was already
            # recorded above; Phase B (the request_camera followup, once it
            # has an image) records the real answer as the one and only
            # assistant turn for this exchange. Recording both would leave
            # two assistant turns with no new user turn in between, and
            # recording this placeholder at all is pure noise once Phase B
            # supersedes it moments later.
            return {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results,
                "think_blocks": think_blocks,
                "scene_description": scene_description,
                "needs_camera": True,
            }

        # request_live_search: same reasoning as request_camera (the browser
        # owns the camera, not the server), but this starts the client's
        # continuous Live Watch loop instead of a one-shot frame. Unlike
        # request_camera there's no Phase B resend of this exact message —
        # the client acknowledges and switches to watching, and the actual
        # finding happens across subsequent is_live_frame ticks — so this
        # response is recorded to memory normally, not suppressed.
        live_search_request = next((r for r in tool_results if r["tool"] == "request_live_search"), None)
        if live_search_request:
            target = live_search_request["arguments"].get("target", "it")
            final_text = self._strip_tool_blocks(clean_text) or f"Watching for {target} now."
            if not is_live_frame:
                self.memory.add("assistant", final_text)
            return {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results,
                "think_blocks": think_blocks,
                "scene_description": scene_description,
                "needs_live_search": True,
                "search_target": target,
            }

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
            # nothing (or, worse, the raw unstripped JSON) — except on a
            # live-frame tick, where log_observation is expected to fire
            # silently on most frames; showing its raw confirmation string
            # would leak bookkeeping into the chat feed exactly where we're
            # trying to suppress noise.
            stripped = self._strip_tool_blocks(clean_text)
            if not stripped and is_live_frame:
                final_text = ""
            else:
                final_text = stripped or "\n".join(r["result"] for r in tool_results)
        else:
            final_text = clean_text or full_text

        # Live-frame turns are allowed to say nothing (the model is told to
        # write SILENT_MARKER when a frame has nothing new relevant to the
        # active goal) — this is the EOS-style silence from the streaming
        # narration problem. Direct user turns never hit this: they always
        # got a real prompt from the user and must always get a real reply,
        # so this check is deliberately gated on is_live_frame.
        if is_live_frame and final_text.strip().upper() == SILENT_MARKER:
            final_text = ""

        # Fold in any timer that finished while we were talking, so a
        # completion surfaces immediately in this reply instead of waiting
        # for the next background /v1/timers/check poll tick.
        timer_update = await self._check_timers_locked()
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

    async def process_stream(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_camera_followup: bool = False,
    ) -> AsyncIterator[dict]:
        """Streaming counterpart to process(), for the Chat & Image UI only —
        not live-frame ticks, which stay on the batched process() path since
        they're mostly silent/one-line and there's nothing worth watching
        stream in. Yields events as the model generates:
          - {"type": "reasoning_delta"/"content_delta", "text": str}
          - {"type": "tool_call_start", "name": str} — the moment a tool call
            is committed to, before its result is known
          - {"type": "tool_result", "tool": str, "result": str}
          - {"type": "done", "data": {...}} — same dict shape process()
            returns, once the turn (including any tool follow-up call) is
            fully resolved.

        Serializes through the same lock as process()/check_timers() — a
        streamed turn still holds it for its whole duration, so a live-frame
        tick or timer poll waits for it to finish rather than interleaving.
        """
        async with self._lock:
            async for event in self._process_stream_locked(image_base64, prompt, is_camera_followup):
                yield event

    async def _stream_backend_call(
        self, image_base64: Optional[str], prompt: str, think: bool, tools: Optional[list[dict]],
    ) -> AsyncIterator[dict]:
        """Delegates to backend.chat_stream() if the backend has one
        (currently only Groq); otherwise falls back to one blocking chat()
        call and reports it as a single "done" event — same contract either
        way, so the caller doesn't need to know which path it got.
        """
        if hasattr(self.backend, "chat_stream"):
            async for event in self.backend.chat_stream(
                image_base64=image_base64,
                prompt=prompt,
                conversation_history=self.memory.get_history()[-10:],
                think=think,
                tools=tools,
            ):
                yield event
        else:
            response = await self.backend.chat(
                image_base64=image_base64,
                prompt=prompt,
                conversation_history=self.memory.get_history()[-10:],
                think=think,
                tools=tools,
            )
            yield {"type": "done", "response": response}

    async def _process_stream_locked(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_camera_followup: bool,
    ) -> AsyncIterator[dict]:
        if not is_camera_followup:
            self.memory.add("user", prompt)

        think = should_think(prompt)
        has_image = bool(image_base64)

        reason_prompt = self._build_reason_prompt(
            prompt=prompt, scene=None, has_image=has_image, think=think, is_live_frame=False,
        )

        native_tools = (
            self.tools.to_openai_tools()
            if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
            else None
        )

        response: Optional[VisionResponse] = None
        async for event in self._stream_backend_call(image_base64, reason_prompt, think, native_tools):
            if event["type"] == "done":
                response = event["response"]
            else:
                yield event

        if response.truncated:
            # Same two-case recovery as _process_locked's non-streaming path
            # (see there for the reasoning). Both retries are one-shot
            # non-streamed calls — truncation is rare enough that streaming
            # the retry too isn't worth the extra complexity — but the
            # recovered text is still surfaced as a content_delta so it
            # appears in the live bubble instead of popping in only at "done".
            if response.reasoning and not response.text.strip():
                logger.warning(
                    "Truncated with reasoning but no answer text — asking "
                    "the model to conclude from its own reasoning (stream)."
                )
                conclude_prompt = (
                    "You were reasoning through this and ran out of space "
                    "before writing your answer. Here is your own reasoning "
                    f"so far:\n\n{response.reasoning}\n\nBased on that, give "
                    "your final answer now — concise, no further reasoning "
                    f"needed.\n\nOriginal question: {prompt}"
                )
                response = await self.backend.chat(
                    image_base64=None, prompt=conclude_prompt, think=False, tools=native_tools,
                )
            else:
                logger.warning(
                    "Truncated with no usable separated reasoning — "
                    "retrying once from scratch with think=False (stream)."
                )
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=None, has_image=has_image, think=think, is_live_frame=False,
                )
                response = await self.backend.chat(
                    image_base64=image_base64, prompt=reason_prompt,
                    conversation_history=self.memory.get_history()[-10:],
                    think=think, tools=native_tools,
                )
            if response.text:
                yield {"type": "content_delta", "text": response.text}

        full_text = response.text

        if response.reasoning:
            think_blocks = [response.reasoning]
            clean_text = full_text.strip()
        else:
            think_blocks = self._extract_think_blocks(full_text)
            clean_text = self._remove_think_blocks(full_text).strip()

        tool_results = []
        if settings.TOOLS_ENABLED:
            if response.tool_calls:
                tool_results = self._run_structured_tool_calls(response.tool_calls)
            else:
                tool_results = await self._execute_tool_calls(clean_text)
            tool_results = [
                r for r in tool_results
                if not r["result"].startswith("Unknown tool:")
                and not r["result"].startswith("JSON parse error:")
                and not r["result"].startswith("Invalid arguments")
            ]
            for r in tool_results:
                yield {"type": "tool_result", "tool": r["tool"], "result": r["result"]}

        camera_request = next((r for r in tool_results if r["tool"] == "request_camera"), None)
        if camera_request:
            final_text = self._strip_tool_blocks(clean_text) or "Let me take a look."
            yield {
                "type": "done",
                "data": {
                    "text": final_text,
                    "model": response.model,
                    "provider": response.provider,
                    "tool_calls": tool_results,
                    "think_blocks": think_blocks,
                    "scene_description": None,
                    "needs_camera": True,
                },
            }
            return

        live_search_request = next((r for r in tool_results if r["tool"] == "request_live_search"), None)
        if live_search_request:
            target = live_search_request["arguments"].get("target", "it")
            final_text = self._strip_tool_blocks(clean_text) or f"Watching for {target} now."
            self.memory.add("assistant", final_text)
            yield {
                "type": "done",
                "data": {
                    "text": final_text,
                    "model": response.model,
                    "provider": response.provider,
                    "tool_calls": tool_results,
                    "think_blocks": think_blocks,
                    "scene_description": None,
                    "needs_live_search": True,
                    "search_target": target,
                },
            }
            return

        if tool_results and any(self.tools.get(r["tool"]).needs_followup for r in tool_results):
            tool_context = "\n\n".join(
                f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
            )
            final_prompt = (
                f"I called tools to answer the user. Here are the results:\n\n"
                f"{tool_context}\n\n"
                f"Original question: {prompt}\n"
                f"Scene context: N/A\n"
                f"Please provide a final answer incorporating these results."
            )
            final_response = await self.backend.chat(image_base64=None, prompt=final_prompt)
            final_text = final_response.text
            if final_text:
                yield {"type": "content_delta", "text": final_text}
        elif tool_results:
            stripped = self._strip_tool_blocks(clean_text)
            final_text = stripped or "\n".join(r["result"] for r in tool_results)
        else:
            final_text = clean_text or full_text

        timer_update = await self._check_timers_locked()
        if timer_update["completed"]:
            timer_lines = "\n".join(
                f"⏰ {t['label']}: {t['message']}" for t in timer_update["completed"]
            )
            final_text = f"{final_text}\n\n{timer_lines}" if final_text else timer_lines
            yield {"type": "content_delta", "text": f"\n\n{timer_lines}"}

        self.memory.add("assistant", final_text)

        yield {
            "type": "done",
            "data": {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results or [],
                "think_blocks": think_blocks,
                "scene_description": None,
            },
        }

    def _parse_retry_after(self, e: Exception, default: float = 5.0) -> float:
        """Best-effort extraction of a provider's suggested wait time from a
        429 error's message text (Groq embeds it as "try again in 6.1s").
        Falls back to `default` if the shape doesn't match — e.g. a
        different backend's error format — so callers always get a usable
        number instead of having to handle None.
        """
        match = re.search(r"try again in ([\d.]+)s", str(e))
        return float(match.group(1)) if match else default

    def _available_tools(self, has_image: bool, is_live_frame: bool) -> list:
        """Tools worth offering for this turn — excludes request_camera/
        request_live_search once there's already an image to look at (a
        live tick or an image-attached message). Shared by both the native
        tool-calling path (native_tools) and the prose tool-list built into
        the prompt for non-native backends, so the two can't drift apart —
        see CLAUDE.md's "second-opinion review" notes on this exact gap.
        """
        offer_camera = not has_image and not is_live_frame
        camera_tool_names = {"request_camera", "request_live_search"}
        return [t for t in self.tools.list_tools() if t.name not in camera_tool_names or offer_camera]

    def _build_reason_prompt(
        self,
        prompt: str,
        scene: Optional[str],
        has_image: bool = False,
        think: bool = True,
        is_live_frame: bool = False,
        strip_task_list: bool = False,
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

        doc_summary = tasklist.render_summary(
            tasklist.get_document(), lean=is_live_frame, observations=not strip_task_list,
        )
        if doc_summary:
            parts.append(f"\n[Task list]\n{doc_summary}")
            parts.append(
                "Indented lines under an item are observations already logged "
                "against it — your memory of what's been seen so far. Check "
                "these before answering instead of assuming only the current "
                "frame/message is all you know."
            )

        parts.append(f"\n[User]\n{prompt}")

        tool_instruction = ""
        if settings.TOOLS_ENABLED:
            # request_camera/request_live_search only make sense when this
            # turn has no image to look at yet — a live-frame ping or an
            # image-attached message already has one, and offering either
            # tool there just invites the model to ask for something it
            # already has (or, for request_live_search, to re-start watching
            # that's already running).
            offer_camera = not has_image and not is_live_frame
            tools = self._available_tools(has_image, is_live_frame)
            tool_list = "\n".join(
                f"- {t.name}({', '.join(t.parameters)}): {t.description}"
                for t in tools
            )
            # Native function-calling backends (Groq — see
            # VisionBackend.SUPPORTS_NATIVE_TOOLS) get tool calls through a
            # structured API field, not by hand-writing JSON into the
            # visible response — telling them to do both would just invite
            # a redundant/malformed text block alongside the real call.
            native = self.backend.SUPPORTS_NATIVE_TOOLS
            format_instruction = (
                ""
                if native
                else (
                    "\n\nYou have tools available. To call one, write this in your "
                    "visible response, not inside a <think> block (only the visible "
                    "response is checked for tool calls):\n"
                    '```tool\n{"name": "tool_name", "arguments": {"arg1": "value"}}\n```\n'
                )
            )
            tool_instruction = (
                format_instruction
                + f"Available tools:\n{tool_list}\n\n"
                "Tool-specific guidance:\n"
                "- start_timer: use for any step that needs waiting (boiling, baking, "
                "marinating, steeping). It runs in the background for free — don't wait "
                "for it or ask about it again yourself; completion is announced to the "
                "user automatically when it's done. Start it, then keep helping with "
                "whatever's next.\n"
                "- update_task_list: use whenever you're guiding a multi-step task (a "
                "recipe, a shopping list, a project)."
                + (
                    " For a plain 'help me find X' with no other steps involved, use "
                    "request_live_search instead — it registers the goal for you."
                    if offer_camera else ""
                )
                + " Do this FIRST, before anything else, so later frames "
                "have a goal to check against. Always send the FULL item list, even items already "
                "completed — anything you leave out is dropped. Mark finished items "
                "'completed' rather than removing them, and use 'skipped' with a note for "
                "substitutions. If a [Task list] is shown above, read it before updating "
                "it, and don't just recite the whole plan back in your reply — the user "
                "can already see it. Each item MUST use the exact key 'content' for its "
                "text — not 'task' or 'label'."
                + (
                    ' Example:\n```tool\n{"name": "update_task_list", "arguments": '
                    '{"title": "Chicken Biryani", "items": [{"content": "Marinate '
                    'chicken", "status": "in_progress"}, {"content": "Cook rice", '
                    '"status": "pending"}]}}\n```\n'
                    if not native else "\n"
                )
                + "- log_observation: call this on every relevant frame for an in-progress "
                "task-list item — e.g. what you currently see related to it — even on "
                "turns where you don't say anything to the user. This is your memory "
                "across frames; a later question like 'where is X' should be answered by "
                "checking these logged notes, not just the current frame.\n"
                + (
                    "- request_camera: no image is attached to this message. If answering "
                    "needs a single look at the current scene, call this instead of "
                    "guessing — do not describe or assume what's currently visible. This "
                    "includes when the user is trying to show you something but hasn't "
                    "attached an image yet (e.g. 'can you see it now', 'here you go') — "
                    "call request_camera to actually prompt them for one, rather than just "
                    "explaining how to attach a photo manually. Telling them how to use "
                    "the interface is only the right answer if they explicitly asked how "
                    "the interface works, not as a substitute for actually looking.\n"
                    "- request_live_search: use when the user wants you to help FIND a "
                    "specific object and one frame won't be enough (they'll need to move "
                    "the camera around while you keep checking). This starts continuous "
                    "watching scoped only to that target — don't use it for general "
                    "cooking help or anything else, that's not enabled through this tool.\n"
                    if offer_camera else ""
                )
            )
        if is_live_frame and doc_summary:
            parts.append(
                f"\nThis is an automated watch tick, not a direct question. Check the "
                f"current frame against the [Task list] item(s) above and their logged "
                f"observations. Always call log_observation with what this frame shows "
                f"relevant to an in-progress item. Only write a visible reply if this "
                f"frame changes something worth telling the user about (progress, a "
                f"problem, the thing they're looking for). If nothing here is new or "
                f"relevant, your entire visible reply must be exactly {SILENT_MARKER} "
                "and nothing else — do not describe the scene."
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

    def _run_structured_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls already parsed by a native-tool-calling backend
        (VisionResponse.tool_calls: [{"id", "name", "arguments"}, ...]) —
        no text-scanning involved, so there's no "wrong field name"/"missing
        arguments wrapper" failure mode to guard against here the way
        _execute_tool_calls has to for the regex path. Still catches
        TypeError for the rarer case of a required argument the model
        genuinely omitted despite the schema declaring it required.
        """
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            tool = self.tools.get(tool_name)
            if not tool:
                results.append({"tool": tool_name, "arguments": arguments, "result": f"Unknown tool: {tool_name}"})
                continue
            try:
                result = tool.fn(**arguments)
            except TypeError as e:
                result = f"Invalid arguments for {tool_name}: {e}"
            results.append({"tool": tool_name, "arguments": arguments, "result": result})
        return results

    async def _execute_tool_calls(self, text: str) -> list[dict]:
        """Find and execute any tool calls in the response text. Fallback
        path for backends without SUPPORTS_NATIVE_TOOLS (see
        _run_structured_tool_calls for the native path).

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
            except json.JSONDecodeError as e:
                results.append({"tool": "unknown", "arguments": {}, "result": f"JSON parse error: {e}"})
                continue

            tool_name = call.get("name")
            # The model sometimes writes {"name": ..., "items": [...]}
            # directly instead of the documented {"name": ..., "arguments":
            # {"items": [...]}} — fall back to everything except "name" so
            # a missing wrapper doesn't turn into a hard TypeError below.
            arguments = call.get("arguments")
            if arguments is None:
                arguments = {k: v for k, v in call.items() if k != "name"}

            tool = self.tools.get(tool_name)
            if not tool:
                results.append({"tool": tool_name, "arguments": arguments, "result": f"Unknown tool: {tool_name}"})
                continue
            try:
                result = tool.fn(**arguments)
            except TypeError as e:
                # Missing/extra/misnamed arguments — surface it as a normal
                # tool result instead of crashing the whole turn.
                result = f"Invalid arguments for {tool_name}: {e}"
            results.append({"tool": tool_name, "arguments": arguments, "result": result})

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
        """Public entry point for the /v1/timers/check poll — acquires the
        turn lock itself, since this is called independently of process()."""
        async with self._lock:
            return await self._check_timers_locked()

    async def _check_timers_locked(self) -> dict:
        """Fire completion calls for any due timers, return completions + free progress.

        Called on every client poll, and opportunistically from process() too
        so an active conversation surfaces a completion immediately instead of
        waiting on the next poll tick. Checking due-ness and computing
        progress is pure arithmetic (timers.due_unfired / timers.active_progress)
        — Groq is only ever called once per timer, right here, only when it's
        actually done. Routed through the same prompt-building and
        tool-execution path as a normal turn, so the completion can also
        update the task list (mark the step done) rather than just narrate it.

        Assumes self._lock is already held by the caller (process() or the
        public check_timers() wrapper) — must not be called concurrently
        with itself, since timers.mark_firing/mark_fired do read-modify-write
        on the same file.
        """
        for t in timers.due_unfired():
            timers.mark_firing(t["id"])
            timer_prompt = (
                f"[Timer completed]\nThe timer '{t['label']}' just finished after "
                f"{t['duration_seconds']} seconds.\nContext: {t['context'] or 'N/A'}\n"
                "Give a brief, practical next-step update for the user. If this step "
                "is tracked in the task list, update it to reflect that it's done."
            )
            reason_prompt = self._build_reason_prompt(
                prompt=timer_prompt, scene=None, has_image=False, think=False,
            )
            native_tools = (
                self.tools.to_openai_tools()
                if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
                else None
            )
            try:
                response = await self.backend.chat(
                    image_base64=None, prompt=reason_prompt, think=False, tools=native_tools,
                )
                clean_text = self._remove_think_blocks(response.text).strip()

                tool_results = []
                if settings.TOOLS_ENABLED:
                    if response.tool_calls:
                        tool_results = self._run_structured_tool_calls(response.tool_calls)
                    else:
                        tool_results = await self._execute_tool_calls(clean_text)
                    tool_results = [
                        r for r in tool_results
                        if not r["result"].startswith("Unknown tool:")
                        and not r["result"].startswith("JSON parse error:")
                        and not r["result"].startswith("Invalid arguments")
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
