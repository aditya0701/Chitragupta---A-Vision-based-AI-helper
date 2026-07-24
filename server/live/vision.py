"""The enriched vision stage — tick captions built for temporal reasoning.

Applies the Qwen3-VL lessons at prompt level (see CLAUDE.md discussion,
2026-07-16):

  tier 1     the previous tick's caption is passed back in as text, so the
             vision model can describe *change and motion* ("door now fills
             most of the frame — user approaching it") instead of an
             isolated snapshot. This is the hosted-API substitute for real
             temporal encodings: the comparison baseline arrives in words.
  action     verbs over nouns — "right hand slicing onion", not
             "hand, knife, onion". Matches how egocentric datasets annotate.
  spatial    consistent place vocabulary, so cross-tick text matching works.
  DeepStack  analog: coarse gist always, fine detail only for goal-relevant
             regions (read labels only when the goal involves finding one).
"""

from __future__ import annotations

from typing import Optional


def build_tick_vision_prompt(
    prev_caption: Optional[str],
    goals: Optional[str] = None,
) -> str:
    parts = [
        "You are the eyes of a live first-person assistant. Describe this camera "
        "frame in at most 4 short sentences, optimized for someone who will read "
        "many of these in sequence to understand what is happening over time.",
        "",
        "Rules:",
        "- Lead with actions and changes, not object inventory. Use action verbs: "
        "'right hand picks up knife, starts slicing onion' — never just 'hand, knife'.",
        "- Include state qualifiers that matter for comparison: near/far, open/closed, "
        "on/off, full/empty, in-hand/put-down, approaching/moving away.",
        "- Name locations consistently and specifically ('top shelf, left side') — "
        "reuse the same words for the same places every time.",
        "- If nothing meaningful changed, say so in one sentence instead of "
        "re-describing the scene.",
        "- Factual only. No advice, no opinions — that is a separate step.",
    ]
    if prev_caption:
        parts += [
            "",
            f"The previous frame (a few seconds ago) was described as: \"{prev_caption}\"",
            "Describe what has CHANGED since then and what action is now in progress. "
            "Note anything that got closer, farther, appeared, or disappeared.",
        ]
    if goals:
        parts += [
            "",
            "Currently relevant goals — if anything in frame relates to these, give "
            "fine-grained detail for it (read labels, count items, exact locations); "
            "keep everything else coarse:",
            goals,
        ]
    return "\n".join(parts)
