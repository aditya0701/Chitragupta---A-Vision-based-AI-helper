"""Gemini backend — uses Google Gemini via Google AI Studio API Key.

Gemini is multimodal, so chat() sends the image and the reasoning prompt
in a single request when both are present — one API call, not two. vision()
(a standalone image-description call) still exists for callers that
specifically want just a description (e.g. the base class's default, or a
backend with genuinely separate vision/reasoning models would use it).

Requires:
  - GEMINI_API_KEY from https://aistudio.google.com/apikey
  - pip install google-generativeai
"""

import base64
import io
from typing import Optional

import google.generativeai as genai
from PIL import Image

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

_REASON_SYSTEM_BASE = (
    "You are Chitragupt, an all-seeing assistant{tools_clause}. "
    "You receive a description of what a camera currently sees, "
    "plus any question from the user.\n\n"
    "Think step by step before responding. "
    "Be concise, practical, and helpful in your final response."
)

_TOOLS_CLAUSE = " with access to tools"

_TOOLS_SECTION = (
    "\n\nAvailable tools:\n"
    "- web_search(query): web search for identifying unknown objects or facts\n"
    "- calculate(expression): arithmetic and unit conversion\n"
    "- get_time(timezone): current time in a given timezone\n\n"
    "Only call a tool if you actually need one to answer — most questions don't. "
    "To call a tool, put this on its own line in your final answer (not just "
    "while thinking about whether to use one):\n"
    "<tool>web_search: query text here</tool>\n"
    "The result will be returned to you automatically."
)

REASON_SYSTEM = _REASON_SYSTEM_BASE.format(
    tools_clause=_TOOLS_CLAUSE if settings.TOOLS_ENABLED else ""
) + (_TOOLS_SECTION if settings.TOOLS_ENABLED else "")


def _decode_image(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))


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
        """Send image to Gemini and return a text description.

        Standalone description-only call — chat() below does NOT use this;
        it attaches the image directly to the reasoning request instead, so
        a normal chat-with-image only costs one API call, not two.
        """
        img = _decode_image(image_base64)

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
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> VisionResponse:
        """Reasoning call. If image_base64 is given, it's attached directly
        to this same request — Gemini sees the image and reasons about it
        in one API call, rather than a separate description call first.
        """
        model = genai.GenerativeModel(self.model_name)

        # Build history from conversation memory
        history = []
        if conversation_history:
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history) if history else model.start_chat()

        content = [REASON_SYSTEM + "\n\n" + prompt]
        if image_base64:
            content.append(_decode_image(image_base64))

        response = await chat.send_message_async(
            content,
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
