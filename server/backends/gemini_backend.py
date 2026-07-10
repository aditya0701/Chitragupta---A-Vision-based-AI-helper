"""Gemini backend — uses Google Gemini via Google AI Studio API Key.

Two-stage pipeline:
  Stage 1 (vision):  Gemini Flash  →  text description of the image
  Stage 2 (reason):  Gemini Flash  →  ReAct reasoning + tool calls + final response

Requires:
  - GEMINI_API_KEY from https://aistudio.google.com/apikey
  - pip install google-generativeai
"""

from typing import Optional

import google.generativeai as genai

from . import VisionBackend, VisionResponse
from ..config import settings

# Gemini calls have no default timeout in the SDK — without one, a stalled
# network path or a slow upstream can hang the request indefinitely instead
# of failing fast with a retryable error.
REQUEST_TIMEOUT_S = 30

VISION_SYSTEM = (
    "Describe everything visible in this image in detail. "
    "Include: objects, people, actions, text, colours, spatial layout, "
    "and anything that might matter for helping someone understand this scene. "
    "Be factual and specific. Do not offer advice or opinions."
)

REASON_SYSTEM = (
    "You are Chitragupt, an all-seeing assistant with access to tools. "
    "You receive a description of what a camera currently sees, "
    "plus any question from the user.\n\n"
    "Think step by step before responding. "
    "Be concise, practical, and helpful in your final response.\n\n"
    "Available tools:\n"
    "- search(query): web search for identifying unknown objects or facts\n"
    "- calculate(expression): arithmetic and unit conversion\n"
    "- translate(text, target_language): translate text visible in the image\n\n"
    "To call a tool, write inside your think block:\n"
    "<tool>search: red mushroom white spots</tool>\n"
    "The result will be returned to you automatically."
)


class GeminiBackend(VisionBackend):
    def __init__(self):
        # The default gRPC transport hangs/times out on some hosting
        # platforms (Render, Vercel, Lambda) whose network egress doesn't
        # play well with long-lived gRPC streams. REST avoids that.
        genai.configure(api_key=settings.GEMINI_API_KEY, transport="rest")
        self.model_name = settings.API_MODEL

    # ── Stage 1: Vision ──────────────────────────────────────────────────────

    async def vision(
        self,
        image_base64: str,
        prompt: str = VISION_SYSTEM,
    ) -> str:
        """Send image to Gemini and return a text description."""
        import base64
        from PIL import Image
        import io

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))

        model = genai.GenerativeModel(self.model_name)
        response = await model.generate_content_async(
            [prompt, img],
            request_options={"timeout": REQUEST_TIMEOUT_S},
        )
        return response.text.strip()

    # ── Stage 2: Reason ──────────────────────────────────────────────────────

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> VisionResponse:
        """Two-stage pipeline: vision -> reasoning.

        If image_base64 is provided, first sends it to Gemini for description,
        then feeds the description into the same model with the user prompt.
        """
        context = ""
        if image_base64:
            context = await self.vision(image_base64)

        model = genai.GenerativeModel(self.model_name)

        # Build history from conversation memory
        history = []
        if conversation_history:
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history) if history else model.start_chat()

        if context:
            user_message = (
                f"[Camera sees]\n{context}\n\n"
                f"[User asks]\n{prompt}"
            )
        else:
            user_message = prompt

        response = await chat.send_message_async(
            [REASON_SYSTEM + "\n\n" + user_message],
            request_options={"timeout": REQUEST_TIMEOUT_S},
        )

        return VisionResponse(
            text=response.text.strip(),
            model=self.model_name,
            provider="gemini",
        )

    # ── Health ───────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            model = genai.GenerativeModel(self.model_name)
            await model.generate_content_async(
                "test",
                request_options={"timeout": REQUEST_TIMEOUT_S},
            )
            return True
        except Exception:
            return False
