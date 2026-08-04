"""LiveAgent — the tick-driven core of the parallel system.

Three entry points, all serialized on one asyncio.Lock (same concurrency
lesson as ChitraguptAgent):

  tick(image)   a live camera frame arrived on the interval. Vision caption
                → doc update → arithmetic triggers → one reasoning call that
                may speak, or replies [SILENT].
  chat(...)     the user typed/said something. Never silent. Doc is the
                shared memory it answers from.
  poll()        no frame, no user — pure trigger arithmetic (free), one
                reasoning call only if something actually fired.

The world doc is loaded at turn start, mutated in memory by tool calls (they
close over `self._doc`), and persisted once at turn end.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from ..agent import ConversationMemory
from ..backends import VisionBackend, should_think
from . import compaction, config, triggers, worlddoc
from .tools import build_live_tools
from .vision import build_tick_vision_prompt

logger = logging.getLogger("chitragupt.live")

SILENT_MARKER = "[SILENT]"

# The counterpart to SILENT: this tick must reach the user NOW, politeness
# budget and all. Reserved for physical risk and for work about to be ruined —
# the cases where a 90-second wait makes the warning worthless.
#
# One flag, one consequence (DECISIONS.md 4.4): this bypasses the politeness
# gate and nothing else. It does not change capture detail, does not affect
# whether questions are asked, and does not itself resolve anything.
URGENT_MARKER = "[URGENT]"

PERSONA = (
    "You are Chitragupta, a live first-person assistant watching through the "
    "user's camera — an attentive record keeper. You track everything in a "
    "persistent world document (shown below) so the user never has to repeat "
    "themselves. Be brief and concrete when you speak."
)


# Text that trails off into something the user was clearly meant to receive
# next — "Here's the plan:", "Do this first —".
_DANGLING_RE = re.compile(r"[:\-–—]\s*$")


def _repair_dangling_plan(text: str, tool_results: list[dict], doc: dict) -> str:
    """Speak the plan when the model announced one and then wrote it into a tool.

    Observed live: the model replied "Got it — we just need the tadka. Here's
    the plan:" and stopped, because the plan itself went into update_tasks.
    update_tasks has needs_followup=False, so no second call happened and the
    naked preamble shipped. The user is LISTENING, not reading — a task list
    that only exists in the doc panel does not reach someone whose hands are in
    the dal, and they had to ask "I cannot see the plan".

    Deterministic rather than another model call: this fires on the exact shape
    of the failure, costs nothing, and cannot itself dangle. Only the first step
    is spoken — reading six steps aloud is how you lose someone at a stove.
    """
    if not text or not _DANGLING_RE.search(text):
        return text
    if not any(r["tool"] in ("update_tasks", "mark_task") for r in tool_results):
        return text
    todo = [t for t in doc["tasks"] if t["status"] in ("pending", "in_progress")]
    if not todo:
        return text
    return f"{text.rstrip(':-–— \t')}: {len(todo)} steps. First: {todo[0]['content']}."


def _fallback_from_work(tool_results: list[dict], doc: dict) -> str:
    """Last resort when a user turn still has no words after the retry.

    Reporting "something went wrong" beside a correctly built plan is worse
    than saying nothing useful — it tells the user to redo work that already
    succeeded, and they have no way to see the doc to know better. If the turn
    demonstrably did something, say what.
    """
    if not any(r["tool"] in ("update_tasks", "mark_task") for r in tool_results or []):
        return ""
    todo = [t for t in doc["tasks"] if t["status"] in ("pending", "in_progress")]
    if not todo:
        return ""
    return f"Plan updated — {len(todo)} steps left. Next: {todo[0]['content']}."


class LiveAgent:
    def __init__(self, backend: VisionBackend):
        self.backend = backend
        self._lock = asyncio.Lock()
        self._doc: Optional[dict] = None  # current turn's doc, seen by tools
        self.tools = build_live_tools(lambda: self._doc)
        self.memory = ConversationMemory(max_turns=30)
        if not backend.SUPPORTS_NATIVE_TOOLS:
            logger.warning(
                "Live system backend %s has no native tool calling — tools disabled "
                "for /v2 (the live system does not carry the old regex-parse path).",
                type(backend).__name__,
            )

    # ── Shared plumbing ──────────────────────────────────────────────────────

    def _native_tools(self) -> Optional[list[dict]]:
        return self.tools.to_openai_tools() if self.backend.SUPPORTS_NATIVE_TOOLS else None

    def _run_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        results = []
        for call in tool_calls or []:
            name = call.get("name")
            arguments = call.get("arguments") or {}
            tool = self.tools.get(name)
            if not tool:
                results.append({"tool": name, "arguments": arguments,
                                "result": f"Unknown tool '{name}'."})
                continue
            try:
                result = tool.fn(**arguments)
            except TypeError as e:
                result = f"Tool '{name}' called with wrong/missing arguments: {e}"
            except Exception as e:
                result = f"Tool '{name}' failed: {e}"
            results.append({"tool": name, "arguments": arguments, "result": str(result),
                            "needs_followup": tool.needs_followup})
        return results

    async def _reason(self, prompt: str, think: bool, history: Optional[list[dict]] = None,
                      require_text: bool = False):
        """One reasoning call + tool execution + at most one follow-up call
        (only when a tool surfaced new information), + one truncation retry.

        `require_text` is the difference between a tick and a user turn. On a
        tick, empty text IS the answer — silence is the default and most ticks
        should produce none. On a user turn it never is: the user asked a
        question and is owed words.

        Without it, a turn that did all its work through tools and said nothing
        reported itself as a failure. Observed live: asked for help with chole,
        the model called update_tasks and set_expectation, wrote a correct
        seven-step plan into the doc, emitted no prose — and the user saw "(no
        reply — something went wrong, try again)" next to a perfectly built
        plan. Neither update_tasks nor set_expectation is flagged
        needs_followup (deliberately: they surface no new information the model
        doesn't already have), so nothing ever asked it to speak.
        """
        response = await self.backend.chat(
            image_base64=None, prompt=prompt,
            conversation_history=history, think=think,
            tools=self._native_tools(),
        )
        if response.truncated:
            logger.info("Live reasoning truncated — retrying once with think=False")
            response = await self.backend.chat(
                image_base64=None, prompt=prompt,
                conversation_history=history, think=False,
                tools=self._native_tools(),
            )

        # A response with neither text nor tool calls is a failed call, not an
        # answer — DeepSeek returns one intermittently. Observed twice while
        # dumping prompts for a real oil-filter planning turn: once on the first
        # call, once on the follow-up. Retry WITH tools, because the turn still
        # needs to do its work; the text-only rescue further down cannot write a
        # plan, so falling straight through to it loses the whole turn's tool
        # calls and the user gets advice that never reaches the world document.
        #
        # Only on user turns. A tick may legitimately produce nothing, and
        # retrying every silent tick would double the cost of the common case.
        if require_text and not (response.text or "").strip() and not response.tool_calls:
            logger.info("Empty reasoning response on a user turn — retrying once with tools")
            response = await self.backend.chat(
                image_base64=None, prompt=prompt,
                conversation_history=history, think=False,
                tools=self._native_tools(),
            )

        tool_results = self._run_tool_calls(response.tool_calls)
        text = (response.text or "").strip()

        followup_results = [r for r in tool_results if r.get("needs_followup")]
        # A user turn that did work but produced no words is a failure, not a
        # silence — go back once for the words, using every result as context.
        speechless = require_text and tool_results and not text
        if speechless and not followup_results:
            followup_results = tool_results

        if followup_results:
            results_text = "\n\n".join(
                f"Result of {r['tool']}:\n{r['result']}" for r in followup_results
            )
            instruction = (
                "[You called tools but said nothing to the user, who is waiting on an "
                "answer and cannot see tool calls or the task list. Here is what you "
                "just did — now SAY it, briefly and out loud. Do not call those tools "
                "again.]"
                if speechless else
                "[You called tools; here are the results — use them to give your final "
                "answer now, without calling those tools again. Anything you already "
                "recorded is recorded: do NOT call update_tasks or set_expectation again "
                "to restate a plan or a watch you just wrote, only if this result "
                "genuinely changes it. Re-recording produces duplicate watches, and every "
                "duplicate is a question asked on every frame for the rest of the "
                "session.]"
            )
            followup_prompt = f"{prompt}\n\n{instruction}\n{results_text}"
            response2 = await self.backend.chat(
                image_base64=None, prompt=followup_prompt,
                conversation_history=history, think=False,
                tools=self._native_tools(),
            )
            extra_results = self._run_tool_calls(response2.tool_calls)
            tool_results.extend(extra_results)
            if (response2.text or "").strip():
                text = response2.text.strip()

        # Last resort for a user turn: still no words after the follow-up.
        #
        # Reproduced live on "walk me through changing an oil filter": the model
        # called web_search, the follow-up call came back completely empty, and
        # the user got "(no reply — something went wrong)" for a turn that was
        # working fine. Checking `speechless` before the follow-up is not
        # enough, because a ReAct chain can spend both calls on tools.
        #
        # tools=None is the whole point. While tools are offered the model can
        # always answer with another call instead of prose; taking them away
        # leaves it nothing to reply with except words. Nothing here can loop —
        # it is one call, and it cannot invoke anything.
        if require_text and not text:
            logger.info("User turn still speechless after follow-up — forcing a text-only reply")
            done = "\n".join(f"- {r['tool']}: {str(r['result'])[:300]}" for r in tool_results)
            final = await self.backend.chat(
                image_base64=None,
                prompt=(
                    f"{prompt}\n\n[You have finished working. Here is what you did"
                    f"{' — nothing yet' if not done else ''}:]\n{done}\n\n"
                    "Now answer the user in plain speech. They are listening and cannot "
                    "see any of the above. Do not call any tools — just say the answer."
                ),
                conversation_history=history, think=False, tools=None,
            )
            if (final.text or "").strip():
                text = final.text.strip()
                response = final

        for r in tool_results:
            r.pop("needs_followup", None)
        return text, tool_results, response

    # ── Tick ─────────────────────────────────────────────────────────────────

    def _goal_hint(self, doc: dict) -> str:
        """Standing context for the vision stage — what this session is about.

        Event-anchored conditions used to be folded in here as 'watch for: X'
        bullets. They are now posed as explicit questions instead; see
        _vision_questions.
        """
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        parts += [t["content"] for t in doc["tasks"] if t["status"] == "in_progress"]
        return "\n".join(f"- {p}" for p in parts)

    def _vision_questions(self, doc: dict, charge: bool = True) -> list[str]:
        """The open event-anchored conditions, as questions for the vision stage.

        These are the reasoning model's own briefs — it cannot see the camera,
        so a separate model looks on its behalf and answers only what it was
        asked. Standing state read straight off the doc, NOT a per-tick round
        trip asking the reasoning model what to look for, which would double the
        model calls on every frame (v1 made the same call — see its watch_for
        notes in agent/agent.py).

        `charge` counts how many times each brief has been asked. Server-managed
        so the model can see a brief going stale and close it: v1's lesson was
        that a focus mode set once is never voluntarily reverted, so something
        has to make the drift visible. Here that pressure is a nudge in the tick
        prompt rather than a hard cutoff, because silently dropping a watch the
        user is still waiting on is the worse failure.

        A full plan registers many watches — a real oil-filter turn produced
        nine, covering steps from chocking the wheels to seating the new
        gasket. Sending all of them every tick is wrong twice over: it is a
        permanent per-tick token tax, and the vision model cannot answer nine
        questions and describe the scene inside one reply, so the answers
        truncate. Only watches relevant NOW are asked — the ones tied to a task
        in progress, plus any not tied to a task, plus high-priority ones,
        which are safety and must never wait their turn. Hard-capped after that.
        """
        in_progress = {t["id"] for t in doc["tasks"] if t["status"] == "in_progress"}
        candidates = []
        for exp in worlddoc.open_expectations(doc):
            if exp["anchor"] != "event" or not (exp.get("condition") or "").strip():
                continue
            relevant = (
                exp["priority"] == "high"
                or not exp.get("task_id")
                or exp["task_id"] in in_progress
            )
            if relevant:
                candidates.append(exp)

        # High priority first, so a safety question is never the one dropped.
        candidates.sort(key=lambda e: 0 if e["priority"] == "high" else 1)
        selected = candidates[: config.MAX_ACTIVE_BRIEFS]
        for exp in selected:
            if charge:
                exp["asks"] = int(exp.get("asks") or 0) + 1
        return [e["condition"].strip() for e in selected]

    def _frame_detail(self, doc: dict) -> str:
        """What resolution the client should use for its NEXT capture.

        Echoed on every /v2 response; the client sizes the next frame from it.
        It has to work one frame ahead because resolution discarded in the
        browser cannot be recovered — a 640px JPEG has already thrown the label
        away by the time it reaches us, and nothing here can upscale it back.
        So the frame that *causes* an upgrade is itself coarse; the brief
        outlives that by many ticks, so the answer is not lost.

        Fine is a consequence of an open question, never a flag the model sets.
        That is the structural fix for v1's failure, where the model set
        detail:'fine' by hand and then never reverted it — see CLAUDE.md's
        MAX_FINE_FRAMES_PER_ITEM. Here, resolving the brief reverts it.

        The asks budget doubles as the cost bound: past MAX_BRIEF_ASKS the
        watch STAYS OPEN but stops buying full resolution. A search that has
        not converged in forty frames will not converge on the forty-first, and
        dropping the watch entirely — v1's other option — silently abandons
        something the user may still be waiting on.

        One flag, one consequence (DECISIONS.md 4.4): this controls capture
        size and nothing else. It must never also gate whether questions are
        asked or whether the tick may speak.
        """
        # 1. The model asked for it. It knows things the doc does not — that a
        #    torque figure is about to be read, that this step turns on seeing
        #    threads rather than gross movement. No rule over doc state
        #    recovers that intent, which is what v1 got right and the first
        #    version of this got wrong: a hands-on hammering task set a focus
        #    and still captured at 640, because "is there an open watch" is a
        #    poor proxy for "does this need resolution".
        if worlddoc.focus_detail(doc) == "fine":
            focus = doc.get("vision_focus") or {}
            if int(focus.get("fine_frames") or 0) < config.MAX_FINE_FOCUS_FRAMES:
                return "fine"

        # 2. A discrete watch is open — reading a label is the case this was
        #    built for, and it cannot be answered from a 640px frame.
        for exp in worlddoc.open_expectations(doc):
            if exp["anchor"] != "event" or not (exp.get("condition") or "").strip():
                continue
            if int(exp.get("asks") or 0) < config.MAX_BRIEF_ASKS:
                return "fine"
        return "coarse"

    def _stale_brief_note(self, doc: dict) -> str:
        """Briefs that have been asked many times without ever being resolved."""
        stale = [
            e for e in worlddoc.open_expectations(doc)
            if e["anchor"] == "event" and int(e.get("asks") or 0) >= config.MAX_BRIEF_ASKS
        ]
        if not stale:
            return ""
        lines = ["", "[Watches going stale — the camera has been asked these many times "
                     "with no resolution. If one no longer applies, or you already have "
                     "your answer, close it with resolve_expectation.]"]
        lines += [f"- ({e['id']}) {e['description']} — asked {e['asks']} times" for e in stale]
        return "\n".join(lines)

    def _build_tick_prompt(self, doc: dict, caption: str, events: list[dict]) -> str:
        lines = [
            PERSONA, "",
            worlddoc.render(doc), "",
            # tick() always add_recent()s the caption before building this, so
            # `recent` is never empty on the live path — but defaulting rather
            # than indexing keeps harnesses and any future caller from
            # exploding on an IndexError deep inside prompt assembly.
            f"[New camera observation, "
            f"{worlddoc.fmt_ts(doc['recent'][-1]['ts']) if doc['recent'] else 'now'}]",
            caption, "",
            "This is an automatic camera tick, NOT a user message. The user is busy; "
            "your default is silence.",
            "",
            "Housekeeping (do silently via tools, this is most of your job):",
            "- If the frame confirms an open expectation happened, call resolve_expectation.",
            "- The observation may open with 'Q1: FOUND / NOT VISIBLE / UNCLEAR' lines. "
            "Those are the camera's direct answers to the watches you set. Treat them as "
            "the answer — do not re-interpret a NOT VISIBLE as a maybe, and do not turn a "
            "generic description into a specific identification. When one is answered for "
            "good, close it with resolve_expectation so the camera stops being asked.",
            "- Check every event-anchored expectation's condition against this frame; if one "
            "fires (the condition is now true), speak up about it.",
            "- If the frame shows where something is kept, call log_environment. Record "
            "only what the observation actually says. It comes from a camera model that "
            "describes generically — 'a bag of lentils', 'a jar of yellow powder' — and "
            "you must not upgrade that into a specific identification the words do not "
            "support. 'Lentils on the shelf' is not 'the black-eyed beans'. If the user "
            "is looking for something specific and you can only see a generic match, say "
            "what you can see and ask them to confirm, rather than announcing a find.",
            "- If a task visibly finished or started, call mark_task.",
            "",
            f"Then: if nothing needs saying to the user, your entire visible reply must be "
            f"exactly {SILENT_MARKER} and nothing else. Speak when: an event-anchored "
            "expectation fired; something genuinely new and important for the active goal "
            "happened; the user is about to make a mistake; a trigger event below asks you "
            "to; something is READY or DONE and they are looking elsewhere; you can see "
            "something they cannot because their attention is on their hands; or a FORM "
            "watch shows the technique going wrong in a way that will cost them the "
            "outcome — say what to change in one sentence, once, and do not nag if they "
            "carry on.",
            "",
            f"SAFETY OVERRIDE. If you can see a physical hazard — a hand in the path of a "
            f"blade, fingers extended flat under a knife, a pan handle turned out over the "
            f"edge, a tool about to slip, a loose fitting under load, something near a "
            f"flame that should not be — say so IMMEDIATELY, in one short sentence, "
            f"starting your reply with {URGENT_MARKER}. That marker skips the politeness "
            f"delay and nothing else, so a warning arrives while it still matters. Lead "
            f"with the action to take ('{URGENT_MARKER} curl your left fingertips back, "
            f"they're flat under the blade'), not with an explanation. Use it only for "
            "physical risk or work about to be ruined — on anything else it is noise, and "
            "noise is how people learn to ignore you.",
        ]
        if events:
            lines += ["", "[Trigger events — these fired by arithmetic while you were away; "
                          "address them in your reply]"]
            lines += [f"- {e['text']}" for e in events]
        stale = self._stale_brief_note(doc)
        if stale:
            lines.append(stale)
        return "\n".join(lines)

    async def tick(self, image_base64: str) -> dict:
        async with self._lock:
            doc = worlddoc.load()
            self._doc = doc
            try:
                prev = worlddoc.last_caption(doc)
                questions = self._vision_questions(doc)
                vision_prompt = build_tick_vision_prompt(
                    prev, self._goal_hint(doc) or None, questions,
                    focus=worlddoc.get_vision_focus(doc))
                caption = await self.backend.vision(
                    image_base64, vision_prompt,
                    max_tokens=config.VISION_MAX_TOKENS
                    + config.VISION_TOKENS_PER_QUESTION * len(questions))

                # This frame was captured at whatever the last response asked
                # for; if the focus is the reason it was fine, charge it.
                worlddoc.charge_focus_frame(doc)

                batch = worlddoc.add_recent(doc, caption)
                if batch:
                    await compaction.compact(self.backend, doc, batch)

                events = triggers.check(doc)
                prompt = self._build_tick_prompt(doc, caption, events)
                text, tool_results, response = await self._reason(prompt, think=False)

                urgent = text.upper().startswith(URGENT_MARKER)
                if urgent:
                    text = text[len(URGENT_MARKER):].lstrip(" :—-")
                if text.upper() == SILENT_MARKER or not text:
                    text = ""
                elif urgent:
                    logger.info("Urgent tick speech — politeness gate bypassed: %r", text[:80])
                else:
                    # Politeness budget: trigger-driven and high-priority speech
                    # always passes; spontaneous commentary waits out the gap.
                    #
                    # The high-priority path used to be unreachable from here,
                    # which cost event-anchored expectations their entire point.
                    # A firing event-anchored expectation is NOT in `events` —
                    # triggers.check() only produces time-anchored and stale-task
                    # events, by design — and may_speak_unprompted() was called
                    # with no priority, so it always evaluated as "normal". A
                    # high-priority watch ("tell me the moment the dal boils
                    # over") therefore got swallowed by the 90s gap unless the
                    # model happened to also call resolve_expectation, which it
                    # has no reason to do when the condition it was guarding
                    # AGAINST just came true. Pass the priority of any still-open
                    # high-priority watch through instead: if one is live and the
                    # model chose to break silence, that is what it is speaking
                    # about. poll() already did this correctly.
                    # in_followup_window: the user asked for something in the
                    # last few minutes, so volunteering an answer is what they
                    # are waiting for, not chatter. Without this the assistant
                    # silences itself for the whole search it just promised to
                    # run — see live/config.py FOLLOWUP_WINDOW_S.
                    important = (
                        bool(events)
                        or triggers.in_followup_window(doc)
                        or any(r["tool"] == "resolve_expectation" for r in tool_results)
                    )
                    watch_priority = "high" if any(
                        e["anchor"] == "event" and e["priority"] == "high"
                        for e in worlddoc.open_expectations(doc)
                    ) else "normal"
                    if not important and not triggers.may_speak_unprompted(doc, watch_priority):
                        logger.info("Politeness gate suppressed unprompted tick speech: %r", text[:80])
                        text = ""
                if text:
                    triggers.mark_spoke(doc)

                worlddoc.save(doc)
                return {
                    "text": text or None,
                    "urgent": bool(urgent and text),
                    "caption": caption,
                    "triggers": [e["text"] for e in events],
                    "tool_calls": tool_results,
                    "model": response.model,
                    "provider": response.provider,
                    "frame_detail": self._frame_detail(doc),
                    "doc": worlddoc.render(doc),
                    "debug": {"vision_prompt": vision_prompt, "reason_prompt": prompt,
                              "raw_text": response.text},
                }
            finally:
                self._doc = None

    # ── Chat ─────────────────────────────────────────────────────────────────

    def _build_chat_prompt(self, doc: dict, user_prompt: str, caption: Optional[str]) -> str:
        lines = [PERSONA, "", worlddoc.render(doc), ""]
        if caption:
            lines += [f"[Current camera frame, {worlddoc.fmt_ts(doc['recent'][-1]['ts'])}]",
                      caption, ""]
        lines += [
            f"[User says] {user_prompt}", "",
            "Answer the user directly — never reply with the silent marker on a user turn. "
            "Use the world document above as your memory: known environment facts answer "
            "'where is X' questions; earlier-session narrative answers 'what happened'. "
            "When you help plan anything with real-world timings, look them up with "
            "web_search if unsure, write the plan with update_tasks, and set_expectation "
            "for each step with a deadline or a watch-for condition — in this same turn, "
            "without being asked. Don't recite the whole plan back; summarize and point "
            "out only what to do first.",
            "",
            "The user is LISTENING to you, not reading. They cannot see your tool calls "
            "or the task list — writing a plan with update_tasks does not show it to "
            "them. So never end on 'here's the plan:' or announce something you then "
            "only put in a tool. Every reply must stand alone as speech: say how many "
            "steps there are and what to do first, in the same breath.",
            "",
            "If the user corrects a fact you recorded or claimed — 'those aren't "
            "lobhiya', 'that's not the toor dal' — call retract_environment_fact "
            "immediately, with what is actually true as the correction. Never leave the "
            "wrong fact in place and just log a new one next to it: both get re-injected "
            "into every later prompt, you will contradict yourself for the rest of the "
            "session, and the user has to keep correcting the same thing. Take the "
            "correction at face value; the user is standing in front of the object and "
            "you are looking at a description of a photograph of it.",
            "",
            "Whenever the task is done BY HAND, call set_vision_focus in the same turn as "
            "the plan, without being asked, and keep it current as the work moves on. One "
            "or two plain sentences saying what the user is physically doing — nothing "
            "more. The standing instructions that make the camera report grip, posture "
            "and danger are attached to every frame automatically; you do not write those. "
            "Do NOT list what it should check and do NOT describe the setup you expect: "
            "you cannot see their kitchen or garage, so an imagined arrangement comes back "
            "as problems that do not exist. Do not create a watch per hazard either — that "
            "produces a dozen overlapping questions and the camera answers none of them "
            "well.",
            "",
            "Set detail='fine' on that focus whenever the step turns on seeing SMALL "
            "things — finger position against a blade, whether a socket sits square, "
            "threads, label text, a gauge. Use detail='coarse' when only gross movement "
            "matters, like stirring or carrying. Fine frames cost about 40% more and are "
            "billed on every frame until you change them, so when the close work is "
            "finished — the part is seated, the cut is done, you have your answer — call "
            "set_vision_focus again with detail='coarse', or with an empty brief if the "
            "hands-on work is over. Reverting is your job; nothing else will do it for "
            "you until a hard frame budget runs out.",
            "",
            "Cover both what could HURT them and what would merely come out BADLY, because "
            "the second is most of the value and the part that gets forgotten. A spanner "
            "pulled instead of pushed, or angled so it rounds the nut. A knife held with "
            "the index finger on the spine, giving uneven slices. A screwdriver not seated "
            "square. A crowded pan that steams instead of browning. None of those will "
            "injure anyone; all of them decide whether the job comes out well, and the "
            "user cannot watch their own hands. Do not restrict this to cooking and do not "
            "wait for a task to look dangerous — any time someone is gripping, turning, "
            "cutting, mixing, seating, aligning or applying force, there is a right way to "
            "hold it that they may not know.",
            "",
            "The camera reports what it sees; judging whether that is dangerous, or merely "
            "wrong, is YOUR job. Ask it for arrangements and positions, never for a "
            "verdict — you will get back things like 'left hand flat, fingertips 2cm from "
            "the blade edge', and it is on you to recognise that as a problem and say so.",
            "",
            "Keep set_expectation for discrete things that either are or are not true yet "
            "— a search ('is the bag of lobhiya visible?'), a state change ('have the "
            "onions gone past golden?'). Those have definite answers. Technique does not.",
            "",
            "Only set a time-anchored expectation for a step the user has actually "
            "STARTED. Deadlines attached to steps they haven't begun go overdue while "
            "they are still gathering ingredients, and you waste their turn cancelling "
            "reminders instead of answering them.",
        ]
        return "\n".join(lines)

    async def chat(self, prompt: str, image_base64: Optional[str] = None) -> dict:
        async with self._lock:
            doc = worlddoc.load()
            self._doc = doc
            try:
                caption = None
                vision_prompt = None
                if image_base64:
                    questions = self._vision_questions(doc)
                    vision_prompt = build_tick_vision_prompt(
                        worlddoc.last_caption(doc), self._goal_hint(doc) or None,
                        questions, focus=worlddoc.get_vision_focus(doc))
                    caption = await self.backend.vision(
                    image_base64, vision_prompt,
                    max_tokens=config.VISION_MAX_TOKENS
                    + config.VISION_TOKENS_PER_QUESTION * len(questions))
                    batch = worlddoc.add_recent(doc, caption)
                    if batch:
                        await compaction.compact(self.backend, doc, batch)

                built = self._build_chat_prompt(doc, prompt, caption)
                text, tool_results, response = await self._reason(
                    built, think=should_think(prompt),
                    history=self.memory.get_history(),
                    require_text=True,  # a user turn is never allowed to be silent
                )
                text = _repair_dangling_plan(text, tool_results, doc)
                if not text:
                    text = (_fallback_from_work(tool_results, doc)
                            or "(no reply — something went wrong, try again)")

                self.memory.add("user", prompt)
                self.memory.add("assistant", text)
                triggers.mark_spoke(doc)      # suppresses stale-task nags
                triggers.mark_user_turn(doc)  # but OPENS the tick follow-up window
                worlddoc.save(doc)
                return {
                    "text": text,
                    "caption": caption,
                    "tool_calls": tool_results,
                    "model": response.model,
                    "provider": response.provider,
                    "frame_detail": self._frame_detail(doc),
                    "doc": worlddoc.render(doc),
                    "debug": {"vision_prompt": vision_prompt, "reason_prompt": built,
                              "raw_text": response.text},
                }
            finally:
                self._doc = None

    # ── Poll (no frame, no user — the free heartbeat) ────────────────────────

    async def poll(self) -> dict:
        async with self._lock:
            doc = worlddoc.load()
            self._doc = doc
            try:
                events = triggers.check(doc)
                # Politeness: overdue expectations are the product working as
                # designed — only 'low' priority ones and stale-task nags wait
                # for the gap.
                speakable = [
                    e for e in events
                    if e["priority"] == "high"
                    or (e["kind"] == "expectation_due" and e["priority"] != "low")
                    or triggers.may_speak_unprompted(doc, e["priority"])
                ]
                if not events:
                    return {"message": None, "doc": worlddoc.render(doc)}
                worlddoc.save(doc)  # persist fired-status even if we stay quiet
                if not speakable:
                    return {"message": None, "doc": worlddoc.render(doc)}

                lines = [
                    PERSONA, "",
                    worlddoc.render(doc), "",
                    "[Trigger events — these just fired by arithmetic; no camera frame, no "
                    "user message. Write ONE short message to the user addressing them. "
                    "Update tasks/expectations via tools as appropriate.]",
                ]
                lines += [f"- {e['text']}" for e in speakable]
                prompt = "\n".join(lines)
                text, tool_results, response = await self._reason(prompt, think=False)
                if text.upper() == SILENT_MARKER:
                    text = ""
                if text:
                    triggers.mark_spoke(doc)
                worlddoc.save(doc)
                return {
                    "message": text or None,
                    "triggers": [e["text"] for e in speakable],
                    "tool_calls": tool_results,
                    "doc": worlddoc.render(doc),
                }
            finally:
                self._doc = None

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        worlddoc.clear()
        self.memory.clear()
