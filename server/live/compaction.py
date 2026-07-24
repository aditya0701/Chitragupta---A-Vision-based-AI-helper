"""Compaction — memory consolidation for the world doc.

When `recent` overflows, the oldest raw tick captions are summarized into
1–3 narrative sentences (time SPANS preserved, not just facts — '12:05–12:40:
rice cooked' keeps temporal structure through the lossy step) plus any
durable environment facts worth promoting. Raw captions are then discarded;
the summary stays. The most recent window is never compacted, so nothing
fresh is ever at risk of being compressed away.

One cheap model call per batch, think=False, no tools.
"""

from __future__ import annotations

import json
import logging
import re

from ..backends import VisionBackend
from . import worlddoc

logger = logging.getLogger("chitragupt.live")

_PROMPT = """You are consolidating an assistant's memory. Below are timestamped
observations from a live camera feed, oldest first. Compress them.

Reply with ONLY a JSON object, no other text:
{{
  "summary": "1-3 sentences of what happened across this period. Mention times or time ranges (HH:MM) for anything with duration or a notable moment.",
  "environment_facts": ["durable facts worth remembering long-term, e.g. where an object is kept — empty list if none"]
}}

Do NOT include transient states (something briefly held or moved) as
environment facts — only things that will still be true and useful later.

Observations:
{observations}"""


async def compact(backend: VisionBackend, doc: dict, batch: list[dict]) -> None:
    """Summarize `batch` (already removed from doc['recent']) into narrative
    + environment facts. On any failure, falls back to keeping a crude
    joined version so raw history is degraded, never silently lost."""
    if not batch:
        return
    obs_text = "\n".join(f"- {worlddoc.fmt_ts(o['ts'])}: {o['text']}" for o in batch)
    start_ts, end_ts = batch[0]["ts"], batch[-1]["ts"]

    try:
        response = await backend.chat(
            image_base64=None,
            prompt=_PROMPT.format(observations=obs_text),
            think=False,
        )
        raw = response.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw)
        summary = (parsed.get("summary") or "").strip()
        facts = parsed.get("environment_facts") or []
        if not summary:
            raise ValueError("empty summary")
    except Exception as e:
        logger.warning(f"Compaction call failed ({e}) — storing crude fallback summary")
        summary = "Uncompacted observations: " + "; ".join(o["text"] for o in batch[-5:])
        facts = []

    worlddoc.add_narrative(doc, start_ts, end_ts, summary)
    for fact in facts:
        if isinstance(fact, str) and fact.strip():
            worlddoc.add_environment_fact(doc, fact)
    logger.info(
        f"Compacted {len(batch)} captions ({worlddoc.fmt_ts(start_ts)}–{worlddoc.fmt_ts(end_ts)}) "
        f"into narrative + {len(facts)} env facts"
    )
