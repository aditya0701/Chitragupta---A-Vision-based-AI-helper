"""Backend factory — returns the correct VisionBackend based on config."""

from . import VisionBackend
from ..config import settings


def get_backend(mode: str | None = None) -> VisionBackend:
    # `mode` override added for the live system (server/live), which picks
    # its backend independently of BACKEND_MODE. Existing callers pass
    # nothing and get the old behavior unchanged.
    mode = mode or settings.BACKEND_MODE

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
        elif provider == "groq":
            from .groq_backend import GroqBackend
            return GroqBackend()
        else:
            raise ValueError(f"Unknown API provider: {provider}")
    elif mode == "local":
        from .ollama_backend import OllamaBackend
        return OllamaBackend()
    elif mode == "hybrid":
        from .deepseek_backend import DeepSeekBackend
        return DeepSeekBackend()
    else:
        raise ValueError(f"Unknown BACKEND_MODE: {mode}")
