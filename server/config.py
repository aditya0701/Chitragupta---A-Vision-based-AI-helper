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

    # Same hybrid split, but with the vision half on DeepInfra's hosted
    # Qwen3-VL rather than Groq. Exists because the v2 live system's tick
    # loop cannot fit under Groq's 8K TPM / 200K TPD free tier at any usable
    # interval — see backends/deepinfra_backend.py for the arithmetic.
    # Selected via LIVE_BACKEND_MODE=deepinfra (live/config.py), which leaves
    # v1's BACKEND_MODE=hybrid on Groq untouched.
    DEEPINFRA_API_KEY: str = os.getenv("DEEPINFRA_API_KEY", "")
    DEEPINFRA_VISION_MODEL: str = os.getenv(
        "DEEPINFRA_VISION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct"
    )

    # Optional. When set, web_search prefers the Brave Search API over the
    # keyless scraped providers — 2,000 queries/month free, and the only
    # provider in that chain with a contract rather than a tolerance behind
    # it. Everything still works without it; see agent/__init__.py's
    # tool_web_search for the fallback order.
    BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")

    # Domains web_search will not return and fetch_page will not retrieve,
    # comma-separated. Suffix-matched on the host, so "wikipedia.org" covers
    # every language subdomain. Set to an empty string to allow everything.
    #
    # Wikipedia is excluded by default for two independent reasons: it is a
    # tertiary source being read aloud as fact on dietary-restriction
    # questions, and it 403s this client anyway, so a Wikipedia hit was a
    # result the model could never follow up with fetch_page.
    #
    # Cost of the default, measured 2026-07-28: Mojeek returned 0 Wikipedia
    # results in 20 on food queries, so the primary provider is unaffected —
    # but 5/5 DuckDuckGo Instant Answer abstracts were Wikipedia, which
    # leaves that last-resort rung mostly empty.
    SEARCH_EXCLUDED_DOMAINS: list[str] = [
        d.strip().lower()
        for d in os.getenv("SEARCH_EXCLUDED_DOMAINS", "wikipedia.org").split(",")
        if d.strip()
    ]

    # The user's local timezone — what get_time answers in when the model
    # doesn't name one, which is almost always, since "what time is it" means
    # local time to the person asking.
    #
    # An IANA zone name, NOT a fixed abbreviation like "CEST". Berlin is CET in
    # winter and CEST in summer; "Europe/Berlin" switches on the right dates by
    # itself, whereas hardcoding either offset is wrong for half the year.
    #
    # Per-user preference is the eventual shape of this (each user setting
    # their own zone in the UI); a single server-wide setting is the stand-in
    # until there is somewhere to store per-user settings at all.
    DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "Europe/Berlin")

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
