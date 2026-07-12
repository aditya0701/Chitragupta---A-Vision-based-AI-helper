"""Backend abstractions for different VLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionResponse:
    text: str
    model: str
    provider: str
    # Populated by backends that return reasoning in a separate structured
    # field (e.g. Groq's reasoning_format="parsed") rather than inline
    # <think> tags mixed into `text`. When set, the agent trusts it directly
    # instead of regex-extracting <think> blocks from `text`.
    reasoning: str = ""


# Keywords that suggest a prompt needs multi-step reasoning rather than a
# quick lookup/greeting-style answer.
_COMPLEXITY_KEYWORDS = (
    "why", "how", "explain", "compare", "difference", "analyze", "analyse",
    "calculate", "reason", "debug", "design", "plan", "trade-off", "tradeoff",
    "pros and cons", "step by step", "should i", "what if",
)


def should_think(prompt: str) -> bool:
    """Heuristic: does this prompt warrant Qwen3 chain-of-thought?

    Short, simple prompts (greetings, quick factual asks) skip thinking for
    speed; longer or clearly analytical/multi-step prompts get it. Backends
    use this to decide whether to send Qwen3's native /think or /no_think
    directive, per the adaptive-thinking design in CLAUDE.md.
    """
    if len(prompt) > 80:
        return True
    lowered = prompt.lower()
    return any(kw in lowered for kw in _COMPLEXITY_KEYWORDS)


class VisionBackend(ABC):
    # True only for backends that genuinely run two separate specialized
    # models (e.g. Colab's qwen3-vl + qwen3) where a distinct vision call is
    # unavoidable. API-mode backends use one multimodal model that can see
    # the image and reason about it in the same request, so this stays
    # False for them — no reason to pay for two calls when one will do.
    SPLIT_VISION_REASONING: bool = False

    @abstractmethod
    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        think: bool = True,
    ) -> VisionResponse:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    async def vision(
        self,
        image_base64: str,
        prompt: str = "Describe everything visible in this image in detail. Include: objects, people, actions, text, colours, spatial layout, and anything that might matter for helping someone understand this scene. Be factual and specific.",
    ) -> str:
        """Stage 1: Describe an image. Returns a text description.

        Default implementation calls chat() with the image.
        Backends with dedicated vision models (e.g. Colab+Ollama) override this.
        """
        response = await self.chat(
            image_base64=image_base64,
            prompt=prompt,
        )
        return response.text
