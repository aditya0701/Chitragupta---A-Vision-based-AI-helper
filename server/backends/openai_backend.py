"""OpenAI backend — uses GPT-4o or any OpenAI vision model."""

import base64
from typing import Optional

from openai import AsyncOpenAI

from . import VisionBackend, VisionResponse
from ..config import settings


class OpenAIBackend(VisionBackend):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.API_MODEL

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> VisionResponse:
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_content = []
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high",
                },
            })
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
        )

        return VisionResponse(
            text=resp.choices[0].message.content,
            model=self.model,
            provider="openai",
        )

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
