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

# How long after a user turn a tick may still speak without waiting out the
# politeness gap. Answering the user used to RESET that gap, which silenced
# exactly the follow-up they were waiting for: asked to find the onions, the
# assistant replied "I'll point them out as soon as they're in view", and that
# reply gagged it for the entire 90s search — it found them at +27s, logged
# them silently, and said nothing until asked again. A recent request is the
# one moment a follow-up is *solicited*, so it opens a window instead.
FOLLOWUP_WINDOW_S: int = int(os.getenv("LIVE_FOLLOWUP_WINDOW_S", "180"))

# How many times an event-anchored brief may be put to the camera before the
# reasoning model is nudged to resolve or cancel it. A nudge, not a cutoff:
# v1's lesson was that a focus mode set once never gets voluntarily reverted,
# but silently dropping a watch the user is still waiting on is worse than
# asking one stale question. At a 6s tick that is roughly four minutes.
MAX_BRIEF_ASKS: int = int(os.getenv("LIVE_MAX_BRIEF_ASKS", "40"))

# How many watches may be put to the camera on a single frame. A whole plan's
# worth of watches is far too many to ask at once — they cannot all be answered
# inside one vision reply, and every one is billed on every tick.
MAX_ACTIVE_BRIEFS: int = int(os.getenv("LIVE_MAX_ACTIVE_BRIEFS", "4"))

# Vision reply budget. The base covers the scene description; each question
# needs room for its own answer line, or the answers truncate mid-block and the
# description never arrives at all.
VISION_MAX_TOKENS: int = int(os.getenv("LIVE_VISION_MAX_TOKENS", "200"))
VISION_TOKENS_PER_QUESTION: int = int(os.getenv("LIVE_VISION_TOKENS_PER_QUESTION", "60"))

# Server-side floor between accepted ticks, same safety-net role as
# LIVE_FRAME_MIN_INTERVAL_S in the old system.
TICK_MIN_INTERVAL_S: float = float(os.getenv("LIVE_TICK_MIN_INTERVAL_S", "1.5"))
