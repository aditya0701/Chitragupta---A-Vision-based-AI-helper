"""Gemini backend — uses Google Gemini vision models."""

from typing import Optional

import google.generativeai as genai

from . import VisionBackend, VisionResponse
from ..config import settings


class GeminiBackend(VisionBackend):
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.API_MODEL)

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> VisionResponse:
        import base64
        from PIL import Image
        import io

        contents = []

        if conversation_history:
            for msg in conversation_history:
                contents.append({"role": msg["role"], "parts": [msg["content"]]})

        parts = []
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            parts.append(img)
        parts.append(prompt)

        # Gemini uses a different chat structure
        chat = self.model.start_chat()
        response = chat.send_message(parts)

        return VisionResponse(
            text=response.text,
            model=settings.API_MODEL,
            provider="gemini",
        )

    async def health_check(self) -> bool:
        try:
            self.model.generate_content("test")
            return True
        except Exception:
            return False
