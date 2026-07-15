"""Configuration loader for Chitragupt server."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class Settings:
    # Backend mode: "colab" | "api" | "local" | "hybrid"
    BACKEND_MODE: str = os.getenv("BACKEND_MODE", "colab")

    # Colab (Ollama on Colab via ngrok)
    COLAB_OLLAMA_URL: str = os.getenv("COLAB_OLLAMA_URL", "")
    COLAB_VISION_MODEL: str = os.getenv("COLAB_VISION_MODEL", "qwen3-vl:8b")
    COLAB_REASON_MODEL: str = os.getenv("COLAB_REASON_MODEL", "qwen3:8b")

    # Legacy colab settings (fallback)
    COLAB_API_URL: str = os.getenv("COLAB_API_URL", "")
    COLAB_API_KEY: str = os.getenv("COLAB_API_KEY", "chitragupt-secret-key")

    # Cloud APIs
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    API_PROVIDER: str = os.getenv("API_PROVIDER", "gemini")
    API_MODEL: str = os.getenv("API_MODEL", "gemini-flash-latest")

    # Hybrid mode: Groq's qwen3.6-27b does vision only (image -> text
    # description), DeepSeek does all reasoning/tool-calling in text only.
    # Split specifically to get reasoning off Groq's combined 8K TPM cap —
    # see CLAUDE.md's "Known constraints" — onto a provider with no
    # comparable per-minute token ceiling. Reuses GROQ_API_KEY above for
    # the vision half.
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

    # Off by default while testing plain API chat — every tool mention (even
    # unresolved/hallucinated ones) risks an extra API call and clutters
    # output. Flip to "true" once ready to re-enable tool use.
    TOOLS_ENABLED: bool = os.getenv("TOOLS_ENABLED", "false").lower() == "true"

    # Ollama (local)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava:13b")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
