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
from typing import Optional

from groq import AsyncGroq
from openai import AsyncOpenAI

from . import VisionBackend, VisionResponse
from ..config import settings

logger = logging.getLogger("chitragupt")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

VISION_PROMPT = (
    "Describe everything visible in this image in detail. Include: objects, "
    "people, actions, text, colours, spatial layout, and anything that "
    "might matter for helping someone understand this scene. Be factual and "
    "specific. Do not offer advice or opinions — that's a separate step."
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

    async def vision(self, image_base64: str, prompt: str = VISION_PROMPT) -> str:
        resp = await self.vision_client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }],
            max_tokens=512,
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

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
