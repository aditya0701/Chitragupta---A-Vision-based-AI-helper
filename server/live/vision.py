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
    focus: Optional[str] = None,
    focus_mode: str = "form",
) -> str:
    parts = []
    if focus and focus_mode == "read":
        # Reading mode. The form block is not merely unhelpful here, it is
        # actively wrong: asked to read a frozen-meal packet, the camera
        # reported "POSTURE AND GRIP — hand position is not visible" while the
        # cooking instructions the user was holding up went untranscribed.
        # Different job, different instructions, and none of the grip wording
        # is loaded at all — which also keeps this block ~40% shorter.
        parts += [
            "WHAT THE USER IS DOING:",
            focus,
            "",
            "You are the live eyes of an AI assistant. It cannot see, and it needs to "
            "READ what is in this frame. Transcribe every piece of legible text you can "
            "see, verbatim.",
            "",
            "  - Copy the words exactly as printed. Do not paraphrase, translate, tidy "
            "up or summarise — whoever reads this needs the original wording.",
            "  - Numbers, times, temperatures, weights and units EXACTLY as shown "
            "('3 ½ MIN', '180°C', '400g'). These are the whole point; an approximated "
            "number is worse than no number.",
            "  - Keep the structure: separate lines stay separate, headings stay with "
            "the text under them, and lists stay in order.",
            "  - Say which language the text is in if it is not English.",
            "",
            "Where you cannot read something, say so precisely and say WHY — too small, "
            "blurred, glare, cut off at the edge, angled away, folded, obscured by a "
            "hand. Then say what would fix it: rotate it upright, hold it closer, "
            "flatten it, tilt it away from the light, move a thumb. The user is holding "
            "the thing and can act on that immediately, but only if you tell them.",
            "",
            "Never guess at a character or a digit to complete a word or number. An "
            "honest 'the second digit is not legible' is far more useful than a "
            "plausible invention, which nobody downstream can detect.",
            "",
            "After the text, add one short line about anything in the frame that looks "
            "unsafe or about to go wrong. Then stop.",
            "",
        ]
    elif focus:
        # The standing lens, as opposed to `questions` which are closed asks.
        #
        # Discrete questions cannot cover technique, because we do not know
        # what the frame will contain when we write them — a checklist composed
        # in advance makes everything not on it invisible. A brief names the job
        # and the dimensions that matter and then gets out of the way, which is
        # what a capable vision model actually needs.
        #
        # The last clause is the part a question list can never provide: an open
        # channel for whatever is wrong that nobody thought to ask about.
        parts += [
            "WHAT THE USER IS DOING:",
            focus,
            "",
            # Everything below this line is STANDARD and identical for every
            # task. Only the line above comes from the reasoning model.
            #
            # The model used to write the whole block, and it wrote checklists:
            # "whether a wheel chock sits behind a rear wheel; whether a drain
            # pan is directly underneath the filter". That enumerates one
            # expected arrangement, so a different but perfectly fine setup
            # reads back as a series of absent items — false findings about a
            # job being done correctly. The reasoning model cannot know the
            # user's actual setup, so it must not be the one describing it.
            "You are the live eyes of an AI assistant helping this person through that "
            "task. They are working with their hands and cannot see themselves. "
            "Alongside anything else you describe, report:",
            "",
            "  POSTURE AND GRIP — how they are holding the tool or the work. Which hand, "
            "where the fingers and thumb sit, where the other hand is, how the tool meets "
            "the material, which direction force is being applied, how the body is braced.",
            "",
            "  DANGER — anything about the arrangement that could injure them. A blade "
            "held or presented wrongly, a hand in the path of a knife or a hammer, fingers "
            "where a tool will go if it slips, a limb under something that could drop, "
            "contact with something hot, sharp, spinning or live.",
            "",
            "Report these as arrangements you can SEE, concretely and physically — which "
            "hand is where, how a tool is held and which way it is turned, what is "
            "touching or near what, what is underneath, how close things are, which way "
            "something points, what is balanced or leaning. State positions, not "
            "conclusions.",
            "",
            "This is NOT a checklist and there is nothing to tick off. If something "
            "mentioned above is simply not in this frame, do not mention it at all — "
            "absence is not a finding, and the user's setup will not match whatever was "
            "imagined when this was written. Describe the arrangement in front of you on "
            "its own terms.",
            "",
            "Do NOT say whether any of it is safe, correct, proper, careful or fine — "
            "that judgement belongs to the reader and a wrong reassurance from you is "
            "the most damaging thing you can write. Describe it; they decide.",
            "",
            "Then, separately: if ANYTHING ELSE in the frame looks out of place, "
            "unstable, spilling, overheating, about to fall or be knocked over, left "
            "running, or otherwise wrong — say so in one sentence, even if it has "
            "nothing to do with the focus above and nobody asked. You are the only one "
            "looking at the whole frame.",
            "",
        ]
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
            # Safety briefs live or die on this. "Is the grip safe?" gets a
            # reassuring guess; "are the fingertips curled back or extended
            # flat?" gets a fact the reasoning model can act on. Same
            # observations-not-judgement rule as above (DECISIONS.md 6.3), but
            # it needs saying for bodies specifically — a model asked about a
            # person's hands reaches for reassurance far more readily than it
            # does when asked about a jar.
            "Some questions are about how something is being HELD or DONE — hands, grip, "
            "body position, how a tool meets the work. For those, describe the physical "
            "arrangement precisely and literally: which fingers are where, curled or "
            "extended, what is in contact with what, which way a handle points, how close "
            "one thing is to another. Never say something looks safe, correct, careful or "
            "fine — that judgement is not yours to make and a wrong reassurance is the "
            "most damaging answer you can give.",
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
