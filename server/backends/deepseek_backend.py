"""DeepSeek backend — hybrid split: Groq's qwen3.6-27b handles vision only
(image -> text description), DeepSeek's API handles all reasoning and tool
calling in text only, the same two-stage shape as ColabBackend but with
Groq/DeepSeek in place of the two Ollama models.

Motivation (see CLAUDE.md's "Known constraints"): Groq's free tier caps
qwen3.6-27b at 8,000 tokens/minute, combined input+output, in one pool
shared by the image, conversation history, tool schemas, and task-list
context. Moving reasoning off Groq onto DeepSeek (no comparable per-minute
token ceiling, 1M context, cheap per-token pricing) means only the vision
call's own small, image-only prompt has to fit under Groq's cap — the
reasoning call is free to carry as much history/task-list state as it needs.

Requires:
  - GROQ_API_KEY (same key GroqBackend uses, for the vision half only)
  - DEEPSEEK_API_KEY from https://platform.deepseek.com/api_keys
  - pip install groq openai   (DeepSeek's API is OpenAI-SDK compatible)
"""

import json
import logging
from typing import AsyncIterator, Optional

from groq import AsyncGroq
from openai import AsyncOpenAI

from . import VisionBackend, VisionResponse
from ..config import settings

logger = logging.getLogger("chitragupt")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Kept deliberately short. Every token here is output the reasoning stage must
# read AND counts against Groq's 8K TPM cap — a verbose "describe everything"
# prompt was producing ~575-token paragraphs per frame, which both slowed the
# vision call and burned the per-minute budget in 2-3 calls (see CLAUDE.md's
# vision-latency finding). A one-line gist is all the reasoning stage needs
# when there's no specific goal; goal-directed frames get the tighter
# object-detection directive built in agent.py instead.
VISION_PROMPT = (
    "In 1-2 short sentences, state only the main objects and what's happening "
    "in this image. No lists, no colours/textures/layout detail, no advice."
)


class DeepSeekBackend(VisionBackend):
    SPLIT_VISION_REASONING = True  # Groq sees the image; DeepSeek never does
    SUPPORTS_NATIVE_TOOLS = True   # DeepSeek's API is OpenAI-compatible function calling

    def __init__(self):
        self.vision_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.vision_model = settings.GROQ_VISION_MODEL
        self.client = AsyncOpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = settings.DEEPSEEK_MODEL

    # ── Stage 1: Vision (Groq) ──────────────────────────────────────────────

    async def vision(self, image_base64: str, prompt: str = VISION_PROMPT, max_tokens: int = 160) -> str:
        resp = await self.vision_client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }],
            # 160 (was 512): a gist or a "FOUND/NOT FOUND: <where>" detection
            # answer needs a fraction of this. The old cap let the model run to
            # ~575 tokens of prose, the single biggest lever on both vision
            # latency and the 8K TPM cap. A short answer that gets cut off is
            # still usable; a 575-token one wastes budget every frame.
            max_tokens=max_tokens,
            # No reasoning needed for a pure description call — this is the
            # smallest possible Groq request for this stage, deliberately
            # kept far under the 8K TPM cap since it never carries history,
            # tool schemas, or task-list context (all of that lives in the
            # DeepSeek reasoning call instead).
            extra_body={"reasoning_effort": "none", "reasoning_format": "parsed"},
        )
        return (resp.choices[0].message.content or "").strip()

    # ── Stage 2: Reason (DeepSeek) ───────────────────────────────────────────

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> VisionResponse:
        # agent.py already ran vision() separately for split-stage backends
        # and folded the description into `prompt` (as "[Camera feed]\n...")
        # before calling here — image_base64 arrives as None, same contract
        # as ColabBackend. DeepSeek's API never sees pixels.
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        create_kwargs = dict(model=self.model, messages=messages, max_tokens=2048)
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"

        resp = await self.client.chat.completions.create(**create_kwargs)
        choice = resp.choices[0]
        message = choice.message
        if getattr(resp, "usage", None):
            logger.info(
                f"DeepSeek usage: prompt={resp.usage.prompt_tokens} "
                f"completion={resp.usage.completion_tokens} "
                f"total={resp.usage.total_tokens} finish_reason={choice.finish_reason}"
            )

        parsed_tool_calls = []
        for tc in (getattr(message, "tool_calls", None) or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek tool call '{tc.function.name}' had unparseable arguments: {e}")
                continue
            parsed_tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        return VisionResponse(
            text=message.content or "",
            model=self.model,
            provider="deepseek",
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
        """Same call as chat(), but yields events as tokens arrive — confirmed
        against DeepSeek's docs that their API streams the same OpenAI-shaped
        way Groq's does (stream=True, chat.completion.chunk deltas, tool
        calls supported mid-stream). Event shapes match GroqBackend.chat_stream
        exactly so agent.py's duck-typed dispatch (hasattr(backend,
        "chat_stream")) picks this up with no caller-side changes. Unlike
        Groq, DeepSeek doesn't expose a separate reasoning field the way
        reasoning_format="parsed" does — deepseek-v4-flash's `content` delta
        carries everything, so no reasoning_delta events are emitted here.
        """
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        create_kwargs = dict(model=self.model, messages=messages, max_tokens=2048, stream=True)
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**create_kwargs)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta

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

        logger.info(f"DeepSeek stream finished: finish_reason={finish_reason}")

        parsed_tool_calls = []
        for slot in tool_calls_acc.values():
            try:
                arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek tool call '{slot['name']}' had unparseable arguments: {e}")
                continue
            parsed_tool_calls.append({"id": slot["id"], "name": slot["name"], "arguments": arguments})

        yield {
            "type": "done",
            "response": VisionResponse(
                text="".join(content_parts),
                model=self.model,
                provider="deepseek",
                truncated=finish_reason == "length",
                tool_calls=parsed_tool_calls,
            ),
        }

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
