"""Backend abstractions for different VLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionResponse:
    text: str
    model: str
    provider: str


class VisionBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        image_base64: Optional[str],
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
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
