"""Settings for the live tick system — kept out of server/config.py so the
old system's config surface is untouched. All env-overridable."""

import os

# Which backend the live system uses, independent of the old system's
# BACKEND_MODE. Defaults to hybrid (Groq vision + DeepSeek reasoning) —
# the tick loop's growing text history is exactly what DeepSeek's prefix
# cache discounts, and it keeps reasoning off Groq's 8K TPM cap.
# Set LIVE_BACKEND_MODE=same to follow the old system's BACKEND_MODE.
LIVE_BACKEND_MODE: str = os.getenv("LIVE_BACKEND_MODE", "hybrid")

# ── World doc bounds ─────────────────────────────────────────────────────────
# Raw tick captions kept verbatim before the oldest are compacted away.
RECENT_MAX: int = int(os.getenv("LIVE_RECENT_MAX", "24"))
# How many of the oldest raw captions each compaction pass consumes.
COMPACT_BATCH: int = int(os.getenv("LIVE_COMPACT_BATCH", "16"))
# Durable environment facts kept (oldest dropped first).
MAX_ENV_FACTS: int = int(os.getenv("LIVE_MAX_ENV_FACTS", "30"))
# Compacted narrative entries kept.
MAX_NARRATIVE: int = int(os.getenv("LIVE_MAX_NARRATIVE", "20"))

# ── Trigger engine ───────────────────────────────────────────────────────────
# An in_progress task with no mention (caption/env/narrative) for this long
# earns an unprompted "still on it?" check-in — the "you forgot the rice"
# trigger when nothing in the frame changed.
STALENESS_S: int = int(os.getenv("LIVE_STALENESS_S", "480"))
# Politeness budget: minimum gap between unprompted (tick/poll-initiated)
# utterances. High-priority expectations bypass it; everything else waits.
MIN_UNPROMPTED_GAP_S: int = int(os.getenv("LIVE_MIN_UNPROMPTED_GAP_S", "90"))

# Server-side floor between accepted ticks, same safety-net role as
# LIVE_FRAME_MIN_INTERVAL_S in the old system.
TICK_MIN_INTERVAL_S: float = float(os.getenv("LIVE_TICK_MIN_INTERVAL_S", "1.5"))
