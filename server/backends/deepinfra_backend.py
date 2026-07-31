"""DeepInfra hybrid backend — the v2 live system's vision half.

Identical to DeepSeekBackend except the vision stage runs on DeepInfra's
hosted Qwen3-VL instead of Groq. Reasoning is inherited unchanged: still
DeepSeek, still text-only, still never sees pixels.

Why this exists — Groq's free tier cannot run the v2 tick loop at all. The
v2 vision call is ~1,350 input + ~90 output tokens (a 1024px frame is ~1,000
image tokens on its own; see server/static/live.js MAX_FRAME_DIM), and Groq
caps qwen3.6-27b at 8,000 tokens/minute combined:

    8,000 TPM / ~1,440 tok per tick  =  5.6 ticks/min  =  one tick per ~11s

v2's default interval is 4s (15 ticks/min), so it runs ~3x over the per-minute
cap and starts taking 429s inside the first minute. The daily cap is worse:
200,000 TPD / 1,440 = ~139 ticks TOTAL per day, which at the tick slider's
slowest setting (15s) is about 35 minutes of watching, once, per day.

v1 survives on Groq because its cadence is slower and most of its turns are
text-only. v2 ticks continuously by design — it is a fundamentally heavier
vision consumer, so it needs a provider with no per-minute token ceiling.
DeepInfra's Qwen3-VL-30B-A3B is ~$0.26 per 1,000 ticks; matching Groq's
entire free daily allowance costs about four cents.

Selected per-system via LIVE_BACKEND_MODE=deepinfra, which is exactly what
that seam was added for — v1 keeps running on Groq's free tier untouched.

Requires:
  - DEEPINFRA_API_KEY   from the DeepInfra dashboard, API Keys
  - DEEPSEEK_API_KEY    inherited, for the reasoning half
  - pip install openai  (DeepInfra's API is OpenAI-SDK compatible)
"""

import logging

from openai import AsyncOpenAI

from .deepseek_backend import VISION_PROMPT, DeepSeekBackend
from ..config import settings

logger = logging.getLogger("chitragupt")

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


class DeepInfraHybridBackend(DeepSeekBackend):
    """Vision on DeepInfra, reasoning on DeepSeek. Both class flags are
    inherited from DeepSeekBackend and stay correct: the vision/reasoning
    split is still real, and DeepSeek still supports native tool calling."""

    def __init__(self):
        # Builds the DeepSeek reasoning client (self.client / self.model) and
        # a Groq vision client we immediately replace — cheap, no network.
        super().__init__()
        # The placeholder counts as unset: .env and .env.example both ship
        # "your-deepinfra-key-here", and switching LIVE_BACKEND_MODE before
        # pasting the real key is the obvious order to get wrong. A 401 on
        # every tick surfaces only as a generic tick error, so fail loudly.
        key = settings.DEEPINFRA_API_KEY
        if not key or key == "your-deepinfra-key-here":
            raise ValueError(
                "DEEPINFRA_API_KEY is not set (or is still the placeholder). "
                "Set a real key in server/.env for local runs, or in the Render "
                "dashboard under Environment for the deployed instance."
            )
        self.vision_client = AsyncOpenAI(
            api_key=settings.DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL
        )
        self.vision_model = settings.DEEPINFRA_VISION_MODEL

    # ── Stage 1: Vision (DeepInfra) ──────────────────────────────────────────

    async def vision(
        self, image_base64: str, prompt: str = VISION_PROMPT, max_tokens: int = 160
    ) -> str:
        resp = await self.vision_client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }],
            max_tokens=max_tokens,
            # NOTE: no extra_body here. DeepSeekBackend.vision() passes
            # {"reasoning_effort": "none", "reasoning_format": "parsed"}, which
            # are Groq-specific — DeepInfra either rejects them or silently
            # ignores them, and neither is worth the ambiguity. Qwen3-VL-*-
            # Instruct is not a reasoning model, so there is nothing to switch
            # off in the first place.
        )
        # Same role the "Groq vision usage:" line plays in the hybrid backend:
        # in this split the vision call is the ONLY image cost, so prompt_tokens
        # here IS the per-frame image bill (the ~330-token text brief is the
        # small part). Watching it tells you directly what frame resolution is
        # costing and whether lowering MAX_FRAME_DIM bought what it should.
        # DeepInfra also returns estimated_cost on usage — logged when present,
        # so a session's real spend is greppable straight out of the logs.
        if getattr(resp, "usage", None):
            cost = getattr(resp.usage, "estimated_cost", None)
            logger.info(
                f"DeepInfra vision usage: prompt={resp.usage.prompt_tokens} "
                f"completion={resp.usage.completion_tokens} "
                f"total={resp.usage.total_tokens} "
                f"finish_reason={resp.choices[0].finish_reason}"
                + (f" est_cost=${cost:.6f}" if cost is not None else "")
            )
        return (resp.choices[0].message.content or "").strip()
