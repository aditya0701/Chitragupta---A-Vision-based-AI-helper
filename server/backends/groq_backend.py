"""Groq backend — fast inference on hosted open models via GroqCloud.

Uses qwen/qwen3.6-27b, which handles both vision and reasoning in a single
multimodal model — no separate vision-stage call needed (SPLIT_VISION_REASONING
stays False, per the base class default).

Requires:
  - GROQ_API_KEY from https://console.groq.com/keys
  - pip install groq
"""

import json
import logging
from typing import AsyncIterator, Optional

from groq import AsyncGroq

from . import VisionBackend, VisionResponse
from ..config import settings

logger = logging.getLogger("chitragupt")


class GroqBackend(VisionBackend):
    # Native function-calling (added 2026-07-13) replaces the fragile
    # prompt-parsed ```tool {...}``` convention for this backend: the model
    # can no longer omit the "arguments" wrapper or invent field names the
    # JSON schema doesn't declare, since the API enforces the shape instead
    # of the model hand-writing JSON into free text.
    SUPPORTS_NATIVE_TOOLS = True

    # This backend also exposes chat_stream() (see below) — agent.py checks
    # for that method's presence (duck-typed, not a class flag) to decide
    # whether a turn can stream live or has to fall back to one blocking
    # chat() call, since streaming is currently Groq-only.

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.API_MODEL

    def _build_create_kwargs(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]],
        think: bool,
        tools: Optional[list[dict]],
    ) -> dict:
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        if image_base64:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        # qwen3-32b thinks by default on Groq. The Qwen "/no_think" text
        # convention isn't honored by Groq's serving stack, so reasoning is
        # gated via the actual `reasoning_effort` API param instead (not yet
        # in this SDK version's typed kwargs, so passed through extra_body).
        reasoning_effort = "default" if think else "none"

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            # This model reasons verbosely when thinking is on — 2048 was too
            # low and let it burn the whole budget mid-thought, cutting off
            # before any visible answer was ever written. Can't just push
            # this arbitrarily high though: this account's Groq tier caps
            # requests at 8000 tokens/minute (TPM) *combined* input+output,
            # so max_tokens has to leave headroom for the prompt itself
            # (which grows with conversation history/task list/tool list).
            max_tokens=4096 if think else 1024,
            extra_body={
                "reasoning_effort": reasoning_effort,
                # "parsed" returns reasoning in its own message.reasoning
                # field instead of inline <think> tags mixed into content —
                # the model doesn't reliably close/scope inline tags, which
                # was leaking raw chain-of-thought into the visible answer.
                "reasoning_format": "parsed",
            },
        )
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        return create_kwargs

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> VisionResponse:
        create_kwargs = self._build_create_kwargs(image_base64, prompt, conversation_history, think, tools)
        resp = await self.client.chat.completions.create(**create_kwargs)

        choice = resp.choices[0]
        message = choice.message
        if getattr(resp, "usage", None):
            logger.info(
                f"Groq usage: prompt={resp.usage.prompt_tokens} "
                f"completion={resp.usage.completion_tokens} "
                f"total={resp.usage.total_tokens} finish_reason={choice.finish_reason}"
            )

        parsed_tool_calls = []
        for tc in (getattr(message, "tool_calls", None) or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"Groq tool call '{tc.function.name}' had unparseable arguments: {e}")
                continue
            parsed_tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        return VisionResponse(
            text=message.content or "",
            model=self.model,
            provider="groq",
            reasoning=getattr(message, "reasoning", None) or "",
            truncated=choice.finish_reason == "length",
            tool_calls=parsed_tool_calls,
        )

    async def chat_stream(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[dict]:
        """Same call as chat(), but yields events as tokens arrive instead of
        waiting for the whole response. Event shapes:
          - {"type": "reasoning_delta", "text": str}
          - {"type": "content_delta", "text": str}
          - {"type": "tool_call_start", "name": str} — fired once, the moment
            a tool call's name is first seen (its arguments are still being
            streamed and aren't valid JSON yet, so callers can announce the
            call but not act on it until "done").
          - {"type": "done", "response": VisionResponse} — the fully
            assembled response, same shape chat() returns, so callers can
            reuse the exact same post-processing (tool execution, truncation
            handling, etc.) regardless of which method was used to get here.

        Tool-call argument fragments arrive keyed by index (per OpenAI's
        streaming tool-call convention, which Groq mirrors) and have to be
        concatenated in full before they're valid JSON — that parse only
        happens once the stream ends, in the same place chat() does it.
        """
        create_kwargs = self._build_create_kwargs(image_base64, prompt, conversation_history, think, tools)
        create_kwargs["stream"] = True

        stream = await self.client.chat.completions.create(**create_kwargs)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta

            reasoning_piece = getattr(delta, "reasoning", None)
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
                yield {"type": "reasoning_delta", "text": reasoning_piece}

            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "content_delta", "text": delta.content}

            for tc in (delta.tool_calls or []):
                slot = tool_calls_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    is_new_name = slot["name"] is None
                    slot["name"] = tc.function.name
                    if is_new_name:
                        yield {"type": "tool_call_start", "name": slot["name"]}
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        # No per-token usage available in streaming mode on this SDK version
        # (stream_options isn't accepted) — only the non-streaming chat()
        # path logs token counts; see CLAUDE.md's Groq TPM cap note.
        logger.info(f"Groq stream finished: finish_reason={finish_reason}")

        parsed_tool_calls = []
        for slot in tool_calls_acc.values():
            try:
                arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Groq tool call '{slot['name']}' had unparseable arguments: {e}")
                continue
            parsed_tool_calls.append({"id": slot["id"], "name": slot["name"], "arguments": arguments})

        yield {
            "type": "done",
            "response": VisionResponse(
                text="".join(content_parts),
                model=self.model,
                provider="groq",
                reasoning="".join(reasoning_parts),
                truncated=finish_reason == "length",
                tool_calls=parsed_tool_calls,
            ),
        }

    async def health_check(self) -> bool:
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
