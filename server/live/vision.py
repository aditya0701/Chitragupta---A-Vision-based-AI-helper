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
    questions: Optional[list[str]] = None,
) -> str:
    parts = []
    if questions:
        # The reasoning model's own brief, asked FIRST and answered explicitly.
        #
        # Without this the brief arrived only as a soft "these are relevant, be
        # detailed" nudge at the end, so nothing was ever actually asked and
        # nothing sharp came back. Observed live: the user wanted black-eyed
        # beans, the caption said "several bags of lentils", and the reasoning
        # model — with no answer to read — upgraded that into "I can see the
        # beans". An inference stood in for an observation because no question
        # was posed.
        #
        # Two rules that are load-bearing, both from v1 (DECISIONS.md 6.3, 6.2):
        # ask for OBSERVATIONS, never judgement — this stage reports, the
        # reasoning stage decides — and make a negative answer an explicit
        # RESULT, since a brief whose "no" looks like silence is how frames get
        # dropped without anyone noticing.
        n = len(questions)
        # The count is stated twice on purpose. Asked with a single question,
        # the model answered "Q1/Q2/Q3", inventing two more to hang the rest of
        # its observations on — harmless to read but it makes the answer block
        # unparseable and pads every caption.
        parts += [
            f"FIRST, answer the {n} question(s) below about this frame. Answer before "
            f"the description, exactly one line each, using the exact labels below. "
            f"There are exactly {n} — never invent extra questions; anything else you "
            f"noticed belongs in the description that follows.",
            "",
        ]
        parts += [f"Q{i}: {q}" for i, q in enumerate(questions, 1)]
        parts += [
            "",
            f"Answer format — exactly {n} line(s), one per question:",
            "  Q<n>: FOUND — <exactly where it is in the frame, plus any label text you can read>",
            "  Q<n>: NOT VISIBLE — <what is in that part of the frame instead>",
            "  Q<n>: UNCLEAR — <what you can make out, and what is blocking a confident answer>",
            "",
            "NOT VISIBLE is a real, useful answer — say it rather than stretching. "
            "Never guess an identification to be helpful: report what is actually "
            "legible or visibly distinctive (label text, shape, colour, markings) and "
            "let the reader decide. Do not judge whether the answer is good news.",
            "",
            "THEN, after the answers, describe the frame as follows.",
            "",
        ]
    parts += [
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
        # Observed live: a whole session of captions read "the camera pulls back
        # and tilts upward", "the camera pushes in close", "the camera angle
        # shifts to the right" — the model narrating the videographer instead of
        # the kitchen. The user is holding the phone and walking around, so the
        # largest frame-to-frame delta genuinely IS camera motion, and the
        # change-framing below points straight at it. Those captions cost full
        # price and carry zero information about the task.
        "- NEVER describe the camera itself. Do not write 'the camera pans/tilts/"
        "zooms/moves', or describe the framing, the angle, or what is cut off at "
        "the edges. The user is holding the camera and knows they moved it. "
        "Report only what is in the scene — objects, people, actions, states. If "
        "the view moved to a new place, name what is now visible there ('pantry "
        "shelf: onions in a wooden bowl, garlic beside them'), never the motion "
        "that got you there.",
    ]
    if prev_caption:
        parts += [
            "",
            f"The previous frame (a few seconds ago) was described as: \"{prev_caption}\"",
            "Describe what has CHANGED in the SCENE since then and what action is now "
            "in progress — objects that appeared or disappeared, states that flipped, "
            "what someone is doing now. A different viewpoint is not a change: if the "
            "camera has simply been moved somewhere else, describe what is there, not "
            "the move.",
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
