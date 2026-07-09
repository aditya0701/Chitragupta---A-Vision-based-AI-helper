"""Ollama backend — connects to a local Ollama instance running a vision model."""

import base64
import httpx
from typing import Optional

from . import VisionBackend, VisionResponse
from ..config import settings


class OllamaBackend(VisionBackend):
    def __init__(self):
        self.host = settings.OLLAMA_HOST.rstrip("/")
        self.model = settings.OLLAMA_MODEL

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> VisionResponse:
        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        user_msg = {"role": "user", "content": prompt}
        if image_base64:
            user_msg["images"] = [image_base64]

        messages.append(user_msg)

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

        return VisionResponse(
            text=data["message"]["content"],
            model=self.model,
            provider="ollama",
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
