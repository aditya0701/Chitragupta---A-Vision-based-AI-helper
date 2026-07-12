"""Anthropic backend — uses Claude 3 vision models."""

import base64
from typing import Optional

from anthropic import AsyncAnthropic

from . import VisionBackend, VisionResponse
from ..config import settings


class AnthropicBackend(VisionBackend):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
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

        user_content = []
        if image_base64:
            image_data = base64.b64decode(image_base64)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                },
            })
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
        )

        return VisionResponse(
            text=resp.content[0].text,
            model=self.model,
            provider="anthropic",
        )

    async def health_check(self) -> bool:
        try:
            await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
