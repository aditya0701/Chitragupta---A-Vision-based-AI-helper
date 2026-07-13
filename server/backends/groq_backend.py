"""Groq backend — fast inference on hosted open models via GroqCloud.

Uses qwen/qwen3.6-27b, which handles both vision and reasoning in a single
multimodal model — no separate vision-stage call needed (SPLIT_VISION_REASONING
stays False, per the base class default).

Requires:
  - GROQ_API_KEY from https://console.groq.com/keys
  - pip install groq
"""

import logging
from typing import Optional

from groq import AsyncGroq

from . import VisionBackend, VisionResponse
from ..config import settings

logger = logging.getLogger("chitragupt")


class GroqBackend(VisionBackend):
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.API_MODEL

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
    ) -> VisionResponse:
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

        resp = await self.client.chat.completions.create(
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

        choice = resp.choices[0]
        message = choice.message
        if getattr(resp, "usage", None):
            logger.info(
                f"Groq usage: prompt={resp.usage.prompt_tokens} "
                f"completion={resp.usage.completion_tokens} "
                f"total={resp.usage.total_tokens} finish_reason={choice.finish_reason}"
            )
        return VisionResponse(
            text=message.content or "",
            model=self.model,
            provider="groq",
            reasoning=getattr(message, "reasoning", None) or "",
            truncated=choice.finish_reason == "length",
        )

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
