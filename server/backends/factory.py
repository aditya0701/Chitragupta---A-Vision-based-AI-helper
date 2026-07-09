"""Backend factory — returns the correct VisionBackend based on config."""

from . import VisionBackend
from ..config import settings


def get_backend() -> VisionBackend:
    mode = settings.BACKEND_MODE

    if mode == "colab":
        from .colab import ColabBackend
        return ColabBackend()
    elif mode == "api":
        provider = settings.API_PROVIDER
        if provider == "openai":
            from .openai_backend import OpenAIBackend
            return OpenAIBackend()
        elif provider == "anthropic":
            from .anthropic_backend import AnthropicBackend
            return AnthropicBackend()
        elif provider == "gemini":
            from .gemini_backend import GeminiBackend
            return GeminiBackend()
        else:
            raise ValueError(f"Unknown API provider: {provider}")
    elif mode == "local":
        from .ollama_backend import OllamaBackend
        return OllamaBackend()
    else:
        raise ValueError(f"Unknown BACKEND_MODE: {mode}")
