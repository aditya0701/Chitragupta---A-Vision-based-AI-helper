"""Groq backend — fast inference on hosted open models via GroqCloud.

Scoped to text-only chat for now. Groq's catalog includes vision-capable
models (e.g. Llama 4 Scout), but image support isn't wired up here yet —
this is deliberately text-only until we've evaluated how the reasoning
stage performs and picked a model, per the plan of testing plain chat
before deciding how to extend this to vision/live-streaming.

Requires:
  - GROQ_API_KEY from https://console.groq.com/keys
  - pip install groq
"""

from typing import Optional

from groq import AsyncGroq

from . import VisionBackend, VisionResponse
from ..config import settings


class GroqBackend(VisionBackend):
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.API_MODEL

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> VisionResponse:
        if image_base64:
            raise NotImplementedError(
                "GroqBackend is text-only for now — image input isn't wired up yet."
            )

        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
        )

        return VisionResponse(
            text=resp.choices[0].message.content,
            model=self.model,
            provider="groq",
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
