"""Colab backend — connects to Ollama running on Google Colab via ngrok.

Uses the two-stage pipeline from CLAUDE.md:
  Stage 1 (vision):  qwen3-vl:8b  →  text description of the image
  Stage 2 (reason):  qwen3:8b     →  ReAct reasoning + tool calls + final response
"""

from typing import Optional

import httpx

from . import VisionBackend, VisionResponse
from ..config import settings

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

VISION_MODEL = "qwen3-vl:8b"
REASON_MODEL = "qwen3:8b"


class ColabBackend(VisionBackend):
    SPLIT_VISION_REASONING = True  # genuinely two separate models

    def __init__(self):
        self.api_url = settings.COLAB_OLLAMA_URL.rstrip("/")
        self.vision_model = settings.COLAB_VISION_MODEL or VISION_MODEL
        self.reason_model = settings.COLAB_REASON_MODEL or REASON_MODEL

    # ── Stage 1: Vision ──────────────────────────────────────────────────────

    async def vision(
        self,
        image_base64: str,
        prompt: str = VISION_SYSTEM,
    ) -> str:
        """Send image to qwen3-vl:8b and return a text description."""
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if "message" not in data or "content" not in data["message"]:
            raise ValueError(f"Unexpected Ollama vision response: {data}")
        return data["message"]["content"].strip()

    # ── Stage 2: Reason ──────────────────────────────────────────────────────

    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> VisionResponse:
        """Two-stage pipeline: vision → reasoning.

        If image_base64 is provided, first calls qwen3-vl:8b to describe it,
        then feeds the description into qwen3:8b along with the user prompt.
        """
        context = ""
        if image_base64:
            context = await self.vision(image_base64)

        # Build messages with system prompt and context
        messages = [{"role": "system", "content": REASON_SYSTEM}]

        if conversation_history:
            messages.extend(conversation_history)

        if context:
            user_message = (
                f"[Camera sees]\n{context}\n\n"
                f"[User asks]\n{prompt}"
            )
        else:
            user_message = prompt

        # qwen3 thinks by default — explicitly gate it per-prompt so simple
        # questions skip the chain-of-thought latency.
        directive = "/think" if think else "/no_think"
        user_message = f"{directive} {user_message}"

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.reason_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if "message" not in data or "content" not in data["message"]:
            raise ValueError(f"Unexpected Ollama chat response: {data}")

        return VisionResponse(
            text=data["message"]["content"].strip(),
            model=f"{self.vision_model}+{self.reason_model}",
            provider="colab",
        )

    # ── Health ───────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.api_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
