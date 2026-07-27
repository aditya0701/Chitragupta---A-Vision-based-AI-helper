"""The main Chitragupt agent — orchestrates vision + tools + memory.

On the Groq API backend, one multimodal model handles both vision and
reasoning in a single call. The two-stage vision/reasoning split (separate
Ollama models) only applies to backends with SPLIT_VISION_REASONING set,
e.g. Colab.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import AsyncIterator, Optional

from ..backends import VisionBackend, VisionResponse, should_think
from ..config import settings
from . import ToolRegistry, ConversationMemory, build_default_tools, timers, tasklist

logger = logging.getLogger("chitragupt")

# Sentinel a live-frame turn writes as its entire visible reply to mean
# "nothing new relevant to the active goal" — stripped before display so it
# never leaks to the user, and never honored on a direct user turn (see
# _process_locked).
SILENT_MARKER = "[SILENT]"

# A chatty reasoning backend (DeepSeek, in the active hybrid config) tends to
# *narrate* its silence — "nothing relevant has changed, staying silent" —
# instead of emitting the bare SILENT_MARKER the prompt asks for. On a live
# tick that narration IS the repetitive-description leak we're suppressing, so
# detection can't rely on an exact marker match. These patterns catch a reply
# whose entire content is a declaration that there's nothing worth saying.
# Gated on is_live_frame + a length cap at the call site so a substantive
# update that merely mentions one of these phrases is never suppressed.
_SILENCE_NARRATION_RE = re.compile(
    r"\b(?:stay(?:ing)?|remain(?:ing)?|keep(?:ing)?|be)\s+silent\b"
    r"|\bsilent\s+as\s+instructed\b"
    r"|\bnothing\s+(?:relevant|new|of\s+note|worth\s+(?:noting|mentioning|saying))\b"
    r"|\bno\s+(?:relevant\s+)?(?:change|update)s?\b"
    r"|\b(?:scene|frame|everything)\s+(?:remains?|is\s+still|stays?)\s+(?:the\s+same|unchanged|blurred)\b",
    re.IGNORECASE,
)

# The "pointed at the list instead of saying the step" non-answer, e.g.
# "start with the in-progress item", "I've updated your task list — start with
# the next pending step". This is the failure the user had to scold the model
# out of. The persona + task-list guidance steer away from it; this regex is
# the enforcement backstop (see _is_list_meta_nonanswer). It matches the
# vocabulary of talking ABOUT the plan's mechanics rather than voicing a step.
_LIST_META_RE = re.compile(
    r"\bin[\s-]?progress\b"
    # "check the list" / "check your list" as well as "checklist" — telling a
    # voice user to consult something they cannot see is the same non-answer
    # whether or not there's an article in the way.
    r"|\btask ?list\b|\bcheck (?:the |your )?list\b|\bto-?do\b"
    r"|\b(?:set[\s-]?up|updated|created|made|put together)\b[^.]{0,40}?\b(?:list|plan|checklist|steps)\b"
    r"|\bstart (?:with|by)\b[^.]{0,30}?\b(?:pending|in[\s-]?progress|item|step|task|list|one|next)\b"
    r"|\bnext\b[^.]{0,20}?\b(?:on|in)\b[^.]{0,10}?\blist\b"
    r"|\bstep(?:s)?\b[^.]{0,15}?\b(?:done|left|completed|remaining|pending|in[\s-]?progress)\b"
    r"|\bmarked\b[^.]{0,15}?\b(?:in[\s-]?progress|completed|pending)\b"
    r"|\bon (?:the|your) list\b"
    r"|\bwhat'?s next\b[^.]{0,15}?\blist\b",
    re.IGNORECASE,
)

# A reply containing a real action imperative is voicing a step, so it's NOT
# the empty "check the list" non-answer even if it also mentions the list in
# passing ("updated the list — now heat the oil and add cumin"). Presence of
# one of these suppresses the backstop, keeping it from re-prompting a reply
# that already did its job.
# Deliberately broad, and safe to keep widening: this regex only ever makes
# _is_list_meta_nonanswer *less* likely to fire (a reply containing any
# imperative is treated as a real answer), so a missing verb causes a needless
# corrective call while an extra one costs nothing. Kept generic rather than
# kitchen-only — the assistant guides any hands-on task, not just cooking.
_ACTION_CUE_RE = re.compile(
    r"\b(?:heat|pre-?heat|add|chop|pour|stir|mix|cut|peel|mince|saut[eé]|fry|"
    r"boil|simmer|place|put|grab|take|turn|drain|soak|mash|whisk|blend|season|"
    r"sprinkle|cover|reduce|flip|roast|grate|crush|knead|roll|bring|set the|"
    # Generic hands-on verbs. Every addition is checked against the list-meta
    # phrasings above for collisions — "check", "start", "look", "find", "let",
    # "keep", "move", "read" and friends are deliberately EXCLUDED because they
    # occur inside the non-answers this is meant to catch ("check the list",
    # "start with the in-progress item"), and matching one there would silently
    # disable the correction.
    r"open|close|lift|hold|press|push|pull|slide|attach|detach|connect|unplug|"
    r"plug|screw|unscrew|tighten|loosen|align|insert|wipe|rinse|scrub|"
    r"fold|hang|carry|rotate|unfold|unwrap|untie|fasten|clamp|drill|hammer|"
    r"sand|paint|glue|tape|weigh|pour out|"
    # Screen/desk work.
    r"click|tap|double-click|paste|unzip|reboot|restart|scan)\b",
    re.IGNORECASE,
)


class FrameBuffer:
    """Rolling buffer of frame descriptions for change detection."""

    def __init__(self, max_frames: int = 10):
        self.frames: list[str] = []
        self.max_frames = max_frames

    def add(self, description: str):
        self.frames.append(description)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    @property
    def last(self) -> Optional[str]:
        return self.frames[-1] if self.frames else None

    @property
    def context(self) -> str:
        """Collapse buffered frames into a single context string."""
        if not self.frames:
            return ""
        return "\n---\n".join(
            f"[Frame {i+1}] {d}" for i, d in enumerate(self.frames)
        )

    def has_changed(self, new_description: str, min_words: int = 10) -> bool:
        """Simple change detection: compare word overlap with last frame."""
        if not self.frames:
            return True

        last_words = set(self.frames[-1].lower().split())
        new_words = set(new_description.lower().split())

        if len(last_words) < min_words or len(new_words) < min_words:
            return True

        jaccard = len(last_words & new_words) / len(last_words | new_words)
        return jaccard < 0.6  # less than 60% overlap = scene changed

    def clear(self):
        self.frames.clear()


class ChitraguptAgent:
    """
    The core agent that:
    1. Takes an image + prompt
    2. For backends with separate vision/reasoning models (SPLIT_VISION_REASONING):
       calls backend.vision() first (Stage 1), then backend.chat() with the
       resulting description (Stage 2) — two calls, only when unavoidable.
       For single multimodal backends: passes the image straight into
       backend.chat() alongside the prompt — one call.
    3. Parses tool calls from the response
    4. Executes tools and returns results
    5. Maintains conversation memory
    """

    def __init__(self, backend: VisionBackend, tools: Optional[ToolRegistry] = None):
        self.backend = backend
        self.tools = tools or build_default_tools()
        self.memory = ConversationMemory()
        self.frame_buffer = FrameBuffer()
        # Serializes every turn (typed chat, live-frame ping, timer
        # completion) so two of them can never interleave reads/writes of
        # the shared task-list document or timer state. Not reentrant —
        # internal callers must use the *_locked variants below, never
        # re-acquire this from within a turn that already holds it.
        self._lock = asyncio.Lock()

    async def process(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_live_frame: bool = False,
        is_camera_followup: bool = False,
    ) -> dict:
        async with self._lock:
            return await self._process_locked(image_base64, prompt, is_live_frame, is_camera_followup)

    def _record_debug_step(
        self,
        steps: list[dict],
        label: str,
        prompt: str,
        has_image: bool,
        think: bool,
        tools: Optional[list[dict]],
        response: VisionResponse,
    ) -> None:
        """Append exactly what this backend.chat() call sent and got back —
        the actual text the model saw (reason_prompt already has the system
        framing folded into it, since this app never uses a separate system
        role) and its raw reply, unfiltered by any of the
        silence/tool-stripping logic downstream. Conversation history isn't
        captured here even though it's sent alongside `prompt` on most
        calls — it's just the last N turns already visible as prior messages
        in the chat itself, so repeating it verbatim on every single step
        would be pure noise rather than new information.
        """
        steps.append({
            "label": label,
            "prompt_sent": prompt,
            "has_image": bool(has_image),
            "think": bool(think),
            "tools_offered": [t["function"]["name"] for t in tools] if tools else [],
            "response_text": response.text,
            "response_reasoning": response.reasoning,
            "truncated": response.truncated,
            "tool_calls_raw": response.tool_calls,
            "model": response.model,
            "provider": response.provider,
        })

    async def _process_locked(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_live_frame: bool = False,
        is_camera_followup: bool = False,
    ) -> dict:
        """Process a user request with optional image.

        `is_live_frame` marks an automated live-streaming ping rather than a
        real user question — its prompt is not recorded in conversation
        memory, so routine "watching" frames don't crowd out real turns.

        `is_camera_followup` marks Phase B of a request_camera round trip —
        the client resending the *same* prompt text now that it has an
        image attached. The original prompt was already recorded when Phase
        A ran (see the request_camera short-circuit below), so recording it
        again here would duplicate the user's utterance in memory.
        """
        if not is_live_frame and not is_camera_followup:
            self.memory.add("user", prompt)

        # Every real backend.chat()/chat_stream() call this turn makes gets
        # recorded here — retries, tool-followup calls, all of it — so the
        # /debug UI can show exactly what the model was sent and what it
        # sent back, in order, without guessing which call produced which
        # visible effect.
        debug_steps: list[dict] = []

        split_stages = image_base64 and self.backend.SPLIT_VISION_REASONING

        # ── Stage 1: Vision (split backends only) ───────────────────────
        # Colab's split qwen3-vl + qwen3 setup and DeepSeekBackend's
        # Groq-vision + DeepSeek-reasoning hybrid both set
        # SPLIT_VISION_REASONING, so this block runs for either — it's
        # skipped only for single-call multimodal backends (Groq API,
        # Gemini, OpenAI, Anthropic), which pass the image straight into
        # the single reasoning call below instead.
        scene_description = None
        vision_prompt = None
        goal_aware = False
        if split_stages:
            scene_description, vision_prompt, goal_aware = await self._run_vision_stage(
                image_base64, debug_steps,
            )

            # The word-overlap "has this changed" heuristic below is a poor
            # fit once the vision answer is a short goal-aware yes/no
            # rather than a full paragraph — two consecutive short answers
            # ("No, still just a counter" / "No, empty shelf") share most
            # of their words by construction, so it would short-circuit
            # almost every goal-aware tick no matter what's actually in
            # frame, including the one that finally says YES. Skip it when
            # goal-aware and let the existing client-side pixel-diff gate
            # plus the model's own [SILENT] protocol do the throttling
            # instead — same reasoning CLAUDE.md documents for why the old
            # _is_relevant_tick() pre-filter was removed entirely.
            # is_live_frame gate: this short-circuit exists to keep an autonomous
            # watch tick quiet, and answering "Scene unchanged — still
            # monitoring" to a question someone actually typed is never right.
            # It was reachable — typing while Live Watch runs posts to /v1/chat
            # with an image and is_live_frame=false, so a question asked about a
            # scene that hadn't moved since the last tick got the monitoring
            # blurb instead of an answer. Same family as the streaming-vision
            # bug: a user turn must always get a real reply.
            if (
                is_live_frame
                and not goal_aware
                and not self.frame_buffer.has_changed(scene_description)
            ):
                self.frame_buffer.add(scene_description)
                return {
                    "text": "👁️ Scene unchanged — still monitoring.",
                    "model": getattr(self.backend, "vision_model", "unknown"),
                    "provider": "groq",
                    "tool_calls": [],
                    "scene_unchanged": True,
                    "scene_description": scene_description,
                    "vision_prompt": vision_prompt,
                    "debug": {"steps": debug_steps},
                }

            self.frame_buffer.add(scene_description)

        # ── Stage 2: Reason ──────────────────────────────────────────────
        # Decide once, from the raw user prompt — not the wrapped reasoning
        # prompt below, which always exceeds any length heuristic once the
        # system framing and scene context are added.
        #
        # Previously this forced think=False for every imageless turn where
        # request_camera was on the table, on the theory that "no image yet"
        # meant "nothing to reason about." That conflated two different
        # things: a live tick's trivial "should I look?" check, and a
        # substantive imageless question like "help me plan chicken
        # biryani" — the one turn where getting the recipe/step breakdown
        # right actually matters most. Forcing shallow reasoning on that
        # turn was backwards. should_think(prompt) alone decides now; the
        # truncation-retry below is the actual safety net for a turn that
        # runs long, applied only when it actually happens rather than
        # preemptively on every imageless question.
        think = should_think(prompt)
        has_image = bool(image_base64) and not split_stages

        # Build the reasoning prompt with scene context
        reason_prompt = self._build_reason_prompt(
            prompt=prompt,
            scene=scene_description,
            has_image=has_image,
            think=think,
            is_live_frame=is_live_frame,
            is_camera_followup=is_camera_followup,
        )

        native_tools = (
            [t.to_openai_tool() for t in self._available_tools(has_image, is_live_frame, is_camera_followup)]
            if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
            else None
        )

        # Live ticks aren't recorded to memory and gain nothing from the
        # last 10 chat turns — the [Task list] block already injected into
        # reason_prompt is the durable state that matters here. Sending
        # full history on every tick was dead weight riding along on
        # exactly the calls most likely to hit the Groq TPM cap.
        history = None if is_live_frame else self.memory.get_history()[-10:]

        try:
            response = await self.backend.chat(
                # Split-stage backends already consumed the image in Stage 1
                # above; single-call backends get it here alongside the prompt.
                image_base64=None if split_stages else image_base64,
                prompt=reason_prompt,
                conversation_history=history,
                think=think,
                tools=native_tools,
            )
            self._record_debug_step(debug_steps, "reason", reason_prompt, has_image, think, native_tools, response)
        except Exception as e:
            status = getattr(e, "status_code", None)
            # 413 ("this one request is too big") and 429 ("you've already
            # spent your rolling per-minute budget, this one just tipped it
            # over") are different failure shapes even though both come from
            # the same Groq TPM cap — a 413 shrinks with a leaner prompt, a
            # 429 doesn't, since the request itself may be perfectly small
            # and just arrived too soon after previous ones (e.g. rapid live
            # ticks while actively searching). Neither should surface the
            # raw provider error to the user.
            if status == 413:
                # Degrade once: drop every observation from the task-list
                # injection (not just completed/skipped ones) and force
                # think=False, then retry. If it fails again, let it raise —
                # one degrade attempt is enough to catch a near-miss, not a
                # systemic sizing problem.
                logger.warning(f"Backend rejected request as too large (413) — retrying with a stripped prompt: {e}")
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=scene_description, has_image=has_image,
                    think=think, is_live_frame=is_live_frame, strip_task_list=True,
                    is_camera_followup=is_camera_followup,
                )
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=None,
                    think=think,
                    tools=native_tools,
                )
                self._record_debug_step(debug_steps, "reason (413 retry, stripped)", reason_prompt, has_image, think, native_tools, response)
            elif status == 429:
                retry_after = self._parse_retry_after(e)
                if is_live_frame:
                    # Don't hold the shared lock waiting out a live tick's
                    # rate limit — another tick comes along in a few seconds
                    # anyway. Surface the wait so the frontend can back its
                    # own polling interval off instead of hammering the same
                    # limit again next tick.
                    logger.warning(f"Rate limited (429) on live tick — skipping this tick, suggested wait {retry_after}s.")
                    return {
                        "text": "",
                        "model": "n/a",
                        "provider": "n/a",
                        "tool_calls": [],
                        "think_blocks": [],
                        "scene_description": None,
                        "rate_limited": True,
                        "retry_after": retry_after,
                        "debug": {"steps": debug_steps, "note": f"429 rate limited, skipped: {e}"},
                    }
                # A direct question deserves an actual answer — wait out the
                # provider's suggested delay once (capped, in case the
                # provider ever reports something unreasonable), then retry,
                # rather than surfacing a raw rate-limit error for something
                # the user is actively waiting on.
                logger.warning(f"Rate limited (429) — waiting {retry_after}s then retrying once.")
                await asyncio.sleep(min(retry_after, 10.0))
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=history,
                    think=think,
                    tools=native_tools,
                )
                self._record_debug_step(debug_steps, "reason (429 retry)", reason_prompt, has_image, think, native_tools, response)
            else:
                raise

        if response.truncated:
            # Generation was cut off by max_tokens before the model finished.
            # Two genuinely different situations need different recoveries:
            if response.reasoning and not response.text.strip():
                # Reasoning finished cleanly (it's sitting in the thinking
                # box), but the model ran out of budget before ever writing
                # the actual answer. Don't throw that reasoning away and
                # re-derive it from scratch — feed it back and ask
                # specifically for the conclusion. Cheaper (short answer
                # only, no re-reasoning) and the answer is grounded in work
                # it already did instead of a fresh low-effort guess.
                logger.warning(
                    "Truncated with reasoning but no answer text — asking "
                    "the model to conclude from its own reasoning."
                )
                conclude_prompt = (
                    "You were reasoning through this and ran out of space "
                    "before writing your answer. Here is your own reasoning "
                    f"so far:\n\n{response.reasoning}\n\nBased on that, give "
                    "your final answer now — concise, no further reasoning "
                    f"needed.\n\nOriginal question: {prompt}"
                )
                response = await self.backend.chat(
                    image_base64=None, prompt=conclude_prompt, think=False, tools=native_tools,
                )
                self._record_debug_step(debug_steps, "conclude-from-reasoning (truncation recovery)", conclude_prompt, False, False, native_tools, response)
            else:
                # Reasoning itself got cut off or never separated cleanly —
                # nothing usable to hand back, so start over with a lower
                # reasoning budget rather than risk the same cutoff twice.
                logger.warning(
                    "Truncated with no usable separated reasoning — "
                    "retrying once from scratch with think=False."
                )
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=scene_description, has_image=has_image,
                    think=think, is_live_frame=is_live_frame,
                    is_camera_followup=is_camera_followup,
                )
                response = await self.backend.chat(
                    image_base64=None if split_stages else image_base64,
                    prompt=reason_prompt,
                    conversation_history=history,
                    think=think,
                    tools=native_tools,
                )
                self._record_debug_step(debug_steps, "reason (truncation retry, fresh)", reason_prompt, has_image, think, native_tools, response)

        full_text = response.text

        if response.reasoning:
            # Backend already separated reasoning from the answer (e.g. Groq's
            # reasoning_format="parsed") — nothing to strip, trust it as-is.
            think_blocks = [response.reasoning]
            clean_text = full_text.strip()
        else:
            # Inline-tag convention (local Ollama/Qwen3): reasoning is mixed
            # into the same text wrapped in <think>...</think>.
            think_blocks = self._extract_think_blocks(full_text)
            clean_text = self._remove_think_blocks(full_text).strip()

        tool_results = []
        if settings.TOOLS_ENABLED:
            if response.tool_calls:
                # Native function-calling — structured, no parsing of the
                # visible text needed at all.
                tool_results = self._run_structured_tool_calls(response.tool_calls)
            else:
                # Fallback for backends without SUPPORTS_NATIVE_TOOLS. Only
                # scan the *visible* response for tool calls, not the raw
                # thinking trace — the model often mentions tool syntax
                # hypothetically while reasoning about whether to use one,
                # and scanning full_text (thinking included) treated that
                # hypothetical mention as a real invocation, triggering a
                # wasted second API call.
                tool_results = await self._execute_tool_calls(clean_text)
            # Unresolved matches (unknown tool name, malformed JSON) aren't
            # worth a costly follow-up call — only resolved tool calls should
            # trigger one.
            tool_results = [
                r for r in tool_results
                if not r["result"].startswith("Unknown tool:")
                and not r["result"].startswith("JSON parse error:")
                and not r["result"].startswith("Invalid arguments")
            ]

        # request_camera can't be resolved server-side — the image lives in
        # the browser. Short-circuit here instead of running the normal
        # tool-result/follow-up flow: tell the client to capture a frame and
        # resend this same question, rather than letting the model guess.
        camera_request = next((r for r in tool_results if r["tool"] == "request_camera"), None)
        if camera_request:
            final_text = self._strip_tool_blocks(clean_text) or "Let me take a look."
            # Not recorded to memory here — this is a provisional holding
            # message, not the real answer. The user's question was already
            # recorded above; Phase B (the request_camera followup, once it
            # has an image) records the real answer as the one and only
            # assistant turn for this exchange. Recording both would leave
            # two assistant turns with no new user turn in between, and
            # recording this placeholder at all is pure noise once Phase B
            # supersedes it moments later.
            return {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results,
                "think_blocks": think_blocks,
                "scene_description": scene_description,
                "vision_prompt": vision_prompt,
                "needs_camera": True,
                "debug": {"steps": debug_steps},
            }

        # request_live_search: same reasoning as request_camera (the browser
        # owns the camera, not the server), but this starts the client's
        # continuous Live Watch loop instead of a one-shot frame. Unlike
        # request_camera there's no Phase B resend of this exact message —
        # the client acknowledges and switches to watching, and the actual
        # finding happens across subsequent is_live_frame ticks — so this
        # response is recorded to memory normally, not suppressed.
        live_search_request = next((r for r in tool_results if r["tool"] == "request_live_search"), None)
        if live_search_request:
            target = live_search_request["arguments"].get("target", "it")
            final_text = self._strip_tool_blocks(clean_text) or f"Watching for {target} now."
            if not is_live_frame:
                self.memory.add("assistant", final_text)
            return {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results,
                "think_blocks": think_blocks,
                "scene_description": scene_description,
                "vision_prompt": vision_prompt,
                "needs_live_search": True,
                "search_target": target,
                "debug": {"steps": debug_steps},
            }

        # log_observation defaults to needs_followup=False (it's a silent
        # side effect on most ticks), but a call asking to be heard means this
        # note is the thing the user is waiting for — force the same follow-up
        # call the needs_followup tools get rather than trusting the model to
        # also have written visible text in the same completion. That trust was
        # the actual bug: tool-calling models routinely return an empty content
        # field alongside a tool call, so a noteworthy tick that only called
        # log_observation went completely silent.
        found_alert = self._speak_alert(tool_results)
        # Deliberately NOT the same condition as found_alert. "Worth saying out
        # loud" and "the search is over, shut the camera off" are unrelated, and
        # wiring them to one flag is what made the camera unusable: found was
        # documented as "important enough to guarantee they're told", so the
        # model set it on ordinary progress notes and the camera closed itself
        # mid-session. Closing only ever follows an actual "Find X" goal being
        # satisfied — tasklist.is_find_goal is the same check that guards the
        # item-completion half in add_observation.
        goal_complete = self._goal_complete(tool_results)
        # A side-effect tool (update_task_list, start_timer) ran but the model
        # wrote no visible text with it — on a direct user turn that leaves the
        # user with a bare "⚡ Used tool" blob and no spoken reply (the "it made
        # a task but never said anything back" bug). Treat it like a fresh tick:
        # feed the tool confirmation back so the model produces a real spoken
        # response about what it set up and what's next. Gated to non-live turns
        # (a live tick updating the task list silently is expected and must stay
        # silent) and only fires when the model actually went silent, so a turn
        # where it already narrated the change costs nothing extra.
        side_effect_silent = bool(
            tool_results
            and not is_live_frame
            and not found_alert
            and not any(self.tools.get(r["tool"]).needs_followup for r in tool_results)
            and not self._strip_tool_blocks(clean_text).strip()
        )
        if tool_results and (found_alert or side_effect_silent or any(self.tools.get(r["tool"]).needs_followup for r in tool_results)):
            tool_context = "\n\n".join(
                f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
            )
            if found_alert:
                # This is the moment the user's been waiting for — a plain
                # "target located" report reads as flat/robotic here in a way
                # it wouldn't on a routine tool result. Ask explicitly for a
                # warm, helper-style delivery instead of the default
                # factual/concise tone.
                final_prompt = (
                    f"You just found what the user was looking for. Here's what you saw:\n\n"
                    f"{tool_context}\n\n"
                    f"Original request: {prompt}\n"
                    f"Scene context: {scene_description or 'N/A'}\n"
                    "Tell them now, like a helpful friend who just spotted it for them — "
                    "warm and a little pleased, not a flat status report. Say what it is "
                    "and where it is, in one or two short spoken sentences."
                )
            elif side_effect_silent:
                # The model set something up (task list / timer) but didn't say
                # anything — prompt it to actually tell the user, out loud.
                final_prompt = (
                    f"You just did the following for the user (these are confirmations "
                    f"that the action succeeded, not new information to look up):\n\n"
                    f"{tool_context}\n\n"
                    f"Original message: {prompt}\n"
                    f"Scene context: {scene_description or 'N/A'}\n"
                    "Now reply out loud in one or two short, natural sentences: tell them "
                    "what you set up and what to do first. Don't recite the whole list back."
                )
            else:
                final_prompt = (
                    f"I called tools to answer the user. Here are the results:\n\n"
                    f"{tool_context}\n\n"
                    f"Original question: {prompt}\n"
                    f"Scene context: {scene_description or 'N/A'}\n"
                    f"Please provide a final answer incorporating these results."
                )
            final_response = await self.backend.chat(
                image_base64=None,
                prompt=final_prompt,
            )
            self._record_debug_step(debug_steps, "tool-result follow-up", final_prompt, False, True, None, final_response)
            final_text = final_response.text
        elif tool_results:
            # Every tool called was a pure side effect (e.g. start_timer,
            # update_task_list) — the model's own surrounding text already
            # says what happened, so skip the paid follow-up call and just
            # strip the raw tool-call syntax out of what's shown to the user.
            # If the model wrote nothing but the tool call itself, fall back
            # to the tool's own confirmation string rather than showing
            # nothing (or, worse, the raw unstripped JSON) — except on a
            # live-frame tick, where log_observation is expected to fire
            # silently on most frames; showing its raw confirmation string
            # would leak bookkeeping into the chat feed exactly where we're
            # trying to suppress noise.
            stripped = self._strip_tool_blocks(clean_text)
            if not stripped and is_live_frame:
                final_text = ""
            else:
                final_text = stripped or "\n".join(r["result"] for r in tool_results)
        else:
            final_text = clean_text or full_text

        # Live-frame turns are allowed to say nothing (the model is told to
        # write SILENT_MARKER when a frame has nothing new relevant to the
        # active goal) — this is the EOS-style silence from the streaming
        # narration problem. Direct user turns never hit this: they always
        # got a real prompt from the user and must always get a real reply,
        # so this check is deliberately gated on is_live_frame.
        if is_live_frame and self._is_silent_live_reply(final_text):
            final_text = ""

        # Enforcement backstop: on a direct turn, if the reply pointed at the
        # list instead of voicing the step, force one corrective call that
        # states the concrete next action. Persona + guidance steer away from
        # this; this guarantees it for the specific failure the user had to
        # scold the model out of.
        if not is_live_frame and self._is_list_meta_nonanswer(final_text):
            corrected = await self._voice_concrete_step(
                prompt, debug_steps, "list-meta non-answer correction",
            )
            if corrected:
                final_text = corrected

        # Fold in any timer that finished while we were talking, so a
        # completion surfaces immediately in this reply instead of waiting
        # for the next background /v1/timers/check poll tick.
        timer_update = await self._check_timers_locked()
        if timer_update["completed"]:
            timer_lines = "\n".join(
                f"⏰ {t['label']}: {t['message']}" for t in timer_update["completed"]
            )
            final_text = f"{final_text}\n\n{timer_lines}" if final_text else timer_lines

        if not is_live_frame:
            self.memory.add("assistant", final_text)

        return {
            "text": final_text,
            "model": response.model,
            "provider": response.provider,
            "tool_calls": tool_results or [],
            "think_blocks": think_blocks,
            "scene_description": scene_description,
            "vision_prompt": vision_prompt,
            # Tells the client it's safe (encouraged, even) to auto-stop Live
            # Watch and close the camera — the find-goal that started this
            # session of continuous polling just got marked "completed" in
            # the task list (see tasklist.add_observation's found= handling).
            # Only a real search satisfies this; a merely noteworthy frame
            # sets found_alert instead and leaves the camera running.
            "goal_complete": goal_complete,
            "debug": {"steps": debug_steps, "timer_completions": timer_update["completed"]},
        }

    async def process_stream(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_camera_followup: bool = False,
    ) -> AsyncIterator[dict]:
        """Streaming counterpart to process(), for the Chat & Image UI only —
        not live-frame ticks, which stay on the batched process() path since
        they're mostly silent/one-line and there's nothing worth watching
        stream in. Yields events as the model generates:
          - {"type": "reasoning_delta"/"content_delta", "text": str}
          - {"type": "tool_call_start", "name": str} — the moment a tool call
            is committed to, before its result is known
          - {"type": "tool_result", "tool": str, "result": str}
          - {"type": "done", "data": {...}} — same dict shape process()
            returns, once the turn (including any tool follow-up call) is
            fully resolved.

        Serializes through the same lock as process()/check_timers() — a
        streamed turn still holds it for its whole duration, so a live-frame
        tick or timer poll waits for it to finish rather than interleaving.
        """
        async with self._lock:
            async for event in self._process_stream_locked(image_base64, prompt, is_camera_followup):
                yield event

    async def _stream_backend_call(
        self, image_base64: Optional[str], prompt: str, think: bool, tools: Optional[list[dict]],
    ) -> AsyncIterator[dict]:
        """Delegates to backend.chat_stream() if the backend has one
        (currently only Groq); otherwise falls back to one blocking chat()
        call and reports it as a single "done" event — same contract either
        way, so the caller doesn't need to know which path it got.
        """
        if hasattr(self.backend, "chat_stream"):
            async for event in self.backend.chat_stream(
                image_base64=image_base64,
                prompt=prompt,
                conversation_history=self.memory.get_history()[-10:],
                think=think,
                tools=tools,
            ):
                yield event
        else:
            response = await self.backend.chat(
                image_base64=image_base64,
                prompt=prompt,
                conversation_history=self.memory.get_history()[-10:],
                think=think,
                tools=tools,
            )
            yield {"type": "done", "response": response}

    async def _process_stream_locked(
        self,
        image_base64: Optional[str],
        prompt: str,
        is_camera_followup: bool,
    ) -> AsyncIterator[dict]:
        if not is_camera_followup:
            self.memory.add("user", prompt)

        think = should_think(prompt)
        debug_steps: list[dict] = []

        # Stage 1, same as _process_locked. Without this the image was passed
        # raw to chat_stream() and dropped on the floor — the reasoning model in
        # the hybrid is text-only and ignores the argument — so every typed turn
        # carrying a frame (an upload, and every request_camera follow-up) was
        # answered blind while the prompt claimed an image was attached. See
        # _run_vision_stage for the full account.
        #
        # Deliberately NOT reusing _process_locked's "scene unchanged"
        # short-circuit: that exists to keep an autonomous watch tick quiet, and
        # a typed question must always get a real answer.
        split_stages = bool(image_base64) and self.backend.SPLIT_VISION_REASONING
        scene_description = None
        vision_prompt = None
        if split_stages:
            scene_description, vision_prompt, _ = await self._run_vision_stage(
                image_base64, debug_steps,
            )
            # Let the client log the vision round trip inline, the same way the
            # non-streaming path does — otherwise the debug wire log jumps
            # straight from the POST to the reasoning output and it looks like
            # DeepSeek saw the picture itself.
            yield {
                "type": "vision",
                "vision_prompt": vision_prompt,
                "scene_description": scene_description,
            }

        # False once the frame has been converted to text: the reasoning model
        # is receiving a description, not pixels, and telling it otherwise makes
        # it answer as though it can see.
        has_image = bool(image_base64) and not split_stages

        reason_prompt = self._build_reason_prompt(
            prompt=prompt, scene=scene_description, has_image=has_image, think=think,
            is_live_frame=False, is_camera_followup=is_camera_followup,
        )

        # Was self.tools.to_openai_tools() — every tool, unconditionally — which
        # is why the stream path (typed chat) kept re-offering request_camera on
        # a camera followup and looped (needs_camera=true again and again, the
        # bug the wire log shows). Mirror the non-stream path: filter through
        # _available_tools so request_camera/request_live_search drop out once
        # this turn already has an image or IS the followup to a camera request.
        native_tools = (
            [t.to_openai_tool() for t in self._available_tools(has_image, False, is_camera_followup)]
            if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
            else None
        )

        response: Optional[VisionResponse] = None
        # image_base64=None once the vision stage has run: the frame is already
        # in the prompt as text, and re-sending pixels to a text-only reasoning
        # model is at best ignored (DeepSeek) and at worst double-billed.
        async for event in self._stream_backend_call(
            None if split_stages else image_base64, reason_prompt, think, native_tools,
        ):
            if event["type"] == "done":
                response = event["response"]
            else:
                yield event
        self._record_debug_step(debug_steps, "reason (stream)", reason_prompt, has_image, think, native_tools, response)

        if response.truncated:
            # Same two-case recovery as _process_locked's non-streaming path
            # (see there for the reasoning). Both retries are one-shot
            # non-streamed calls — truncation is rare enough that streaming
            # the retry too isn't worth the extra complexity — but the
            # recovered text is still surfaced as a content_delta so it
            # appears in the live bubble instead of popping in only at "done".
            if response.reasoning and not response.text.strip():
                logger.warning(
                    "Truncated with reasoning but no answer text — asking "
                    "the model to conclude from its own reasoning (stream)."
                )
                conclude_prompt = (
                    "You were reasoning through this and ran out of space "
                    "before writing your answer. Here is your own reasoning "
                    f"so far:\n\n{response.reasoning}\n\nBased on that, give "
                    "your final answer now — concise, no further reasoning "
                    f"needed.\n\nOriginal question: {prompt}"
                )
                response = await self.backend.chat(
                    image_base64=None, prompt=conclude_prompt, think=False, tools=native_tools,
                )
                self._record_debug_step(debug_steps, "conclude-from-reasoning (truncation recovery, stream)", conclude_prompt, False, False, native_tools, response)
            else:
                logger.warning(
                    "Truncated with no usable separated reasoning — "
                    "retrying once from scratch with think=False (stream)."
                )
                think = False
                reason_prompt = self._build_reason_prompt(
                    prompt=prompt, scene=None, has_image=has_image, think=think,
                    is_live_frame=False, is_camera_followup=is_camera_followup,
                )
                response = await self.backend.chat(
                    image_base64=image_base64, prompt=reason_prompt,
                    conversation_history=self.memory.get_history()[-10:],
                    think=think, tools=native_tools,
                )
                self._record_debug_step(debug_steps, "reason (truncation retry, fresh, stream)", reason_prompt, has_image, think, native_tools, response)
            if response.text:
                yield {"type": "content_delta", "text": response.text}

        full_text = response.text

        if response.reasoning:
            think_blocks = [response.reasoning]
            clean_text = full_text.strip()
        else:
            think_blocks = self._extract_think_blocks(full_text)
            clean_text = self._remove_think_blocks(full_text).strip()

        tool_results = []
        if settings.TOOLS_ENABLED:
            if response.tool_calls:
                tool_results = self._run_structured_tool_calls(response.tool_calls)
            else:
                tool_results = await self._execute_tool_calls(clean_text)
            tool_results = [
                r for r in tool_results
                if not r["result"].startswith("Unknown tool:")
                and not r["result"].startswith("JSON parse error:")
                and not r["result"].startswith("Invalid arguments")
            ]
            for r in tool_results:
                yield {"type": "tool_result", "tool": r["tool"], "result": r["result"]}

        camera_request = next((r for r in tool_results if r["tool"] == "request_camera"), None)
        if camera_request:
            final_text = self._strip_tool_blocks(clean_text) or "Let me take a look."
            yield {
                "type": "done",
                "data": {
                    "text": final_text,
                    "model": response.model,
                    "provider": response.provider,
                    "tool_calls": tool_results,
                    "think_blocks": think_blocks,
                    "scene_description": scene_description,
                    "vision_prompt": vision_prompt,
                    "needs_camera": True,
                    "debug": {"steps": debug_steps},
                },
            }
            return

        live_search_request = next((r for r in tool_results if r["tool"] == "request_live_search"), None)
        if live_search_request:
            target = live_search_request["arguments"].get("target", "it")
            final_text = self._strip_tool_blocks(clean_text) or f"Watching for {target} now."
            self.memory.add("assistant", final_text)
            yield {
                "type": "done",
                "data": {
                    "text": final_text,
                    "model": response.model,
                    "provider": response.provider,
                    "tool_calls": tool_results,
                    "think_blocks": think_blocks,
                    "scene_description": scene_description,
                    "vision_prompt": vision_prompt,
                    "needs_live_search": True,
                    "search_target": target,
                    "debug": {"steps": debug_steps},
                },
            }
            return

        # Same two signals as the non-streaming path (_process_locked), kept
        # deliberately separate: "say this out loud" vs "the search is over,
        # close the camera". A find-goal can complete on a typed turn too (the
        # user asks "is it this one?" while watching), which is rendered through
        # this path.
        found_alert = self._speak_alert(tool_results)
        goal_complete = self._goal_complete(tool_results)
        # Same "side-effect tool ran but the model said nothing" recovery as
        # the non-stream path (_process_locked): a typed turn that only calls
        # update_task_list / start_timer and writes no visible text would
        # otherwise show a bare tool blob with no spoken reply. Feed the
        # confirmation back so the model actually tells the user what it set up.
        # Only fires when the model went silent, so normal narrated turns cost
        # nothing extra. (The stream path is never a live-frame tick.)
        side_effect_silent = bool(
            tool_results
            and not found_alert
            and not any(self.tools.get(r["tool"]).needs_followup for r in tool_results)
            and not self._strip_tool_blocks(clean_text).strip()
        )
        if tool_results and (side_effect_silent or any(self.tools.get(r["tool"]).needs_followup for r in tool_results)):
            tool_context = "\n\n".join(
                f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
            )
            if side_effect_silent:
                final_prompt = (
                    f"You just did the following for the user (these are confirmations "
                    f"that the action succeeded, not new information to look up):\n\n"
                    f"{tool_context}\n\n"
                    f"Original message: {prompt}\n"
                    "Now reply out loud in one or two short, natural sentences: tell them "
                    "what you set up and what to do first. Don't recite the whole list back."
                )
            else:
                final_prompt = (
                    f"I called tools to answer the user. Here are the results:\n\n"
                    f"{tool_context}\n\n"
                    f"Original question: {prompt}\n"
                    f"Scene context: N/A\n"
                    f"Please provide a final answer incorporating these results."
                )
            final_response = await self.backend.chat(image_base64=None, prompt=final_prompt)
            self._record_debug_step(debug_steps, "tool-result follow-up (stream)", final_prompt, False, True, None, final_response)
            final_text = final_response.text
            if final_text:
                yield {"type": "content_delta", "text": final_text}
        elif tool_results:
            stripped = self._strip_tool_blocks(clean_text)
            final_text = stripped or "\n".join(r["result"] for r in tool_results)
        else:
            final_text = clean_text or full_text

        # Enforcement backstop — same as the non-stream path (_process_locked):
        # a reply that pointed at the list instead of voicing the step gets one
        # corrective call. Gated to backends whose "stream" is really a single
        # blocking call (no chat_stream — the active DeepSeek hybrid), so
        # nothing was shown to the user incrementally yet and replacing the text
        # is clean. A genuinely token-streaming backend (Groq) would have
        # already surfaced the meta reply, so we don't retro-correct there and
        # lean on the persona/guidance instead.
        if (
            not hasattr(self.backend, "chat_stream")
            and self._is_list_meta_nonanswer(final_text)
        ):
            corrected = await self._voice_concrete_step(
                prompt, debug_steps, "list-meta non-answer correction (stream)",
            )
            if corrected:
                final_text = corrected
                yield {"type": "content_delta", "text": corrected}

        timer_update = await self._check_timers_locked()
        if timer_update["completed"]:
            timer_lines = "\n".join(
                f"⏰ {t['label']}: {t['message']}" for t in timer_update["completed"]
            )
            final_text = f"{final_text}\n\n{timer_lines}" if final_text else timer_lines
            yield {"type": "content_delta", "text": f"\n\n{timer_lines}"}

        self.memory.add("assistant", final_text)

        yield {
            "type": "done",
            "data": {
                "text": final_text,
                "model": response.model,
                "provider": response.provider,
                "tool_calls": tool_results or [],
                "think_blocks": think_blocks,
                "scene_description": scene_description,
                "vision_prompt": vision_prompt,
                "goal_complete": goal_complete,
                "debug": {"steps": debug_steps, "timer_completions": timer_update["completed"]},
            },
        }

    def _parse_retry_after(self, e: Exception, default: float = 5.0) -> float:
        """Best-effort extraction of a provider's suggested wait time from a
        429 error's message text (Groq embeds it as "try again in 6.1s").
        Falls back to `default` if the shape doesn't match — e.g. a
        different backend's error format — so callers always get a usable
        number instead of having to handle None.
        """
        match = re.search(r"try again in ([\d.]+)s", str(e))
        return float(match.group(1)) if match else default

    @staticmethod
    def _speak_alert(tool_results: list[dict]) -> bool:
        """Did an observation this turn ask to be said out loud?

        alert=true is the plain "tell them this now" flag. found=true implies
        it — spotting the thing being searched for is always worth saying —
        so a model that only sets found still gets a spoken reply.
        """
        return any(
            r["tool"] == "log_observation"
            and (r["arguments"].get("alert") or r["arguments"].get("found"))
            for r in tool_results
        )

    @staticmethod
    def _goal_complete(tool_results: list[dict]) -> bool:
        """Did a *search* finish this turn — i.e. is it safe to close the camera?

        Narrower than _speak_alert on purpose. Requires both found=true AND the
        item actually being a "Find X" goal, mirroring the guard in
        tasklist.add_observation that decides whether to complete the item. A
        found=true aimed at an ordinary step now neither completes the step nor
        closes the camera; it just gets spoken.
        """
        return any(
            r["tool"] == "log_observation"
            and r["arguments"].get("found")
            and tasklist.is_find_goal(str(r["arguments"].get("item", "")))
            for r in tool_results
        )

    async def _run_vision_stage(
        self, image_base64: str, debug_steps: list[dict],
    ) -> tuple[str, str, bool]:
        """Stage 1 for split backends: turn the frame into text.

        Extracted so the streaming path can run it too. It could not before,
        and the consequence was severe: _process_stream_locked passed the raw
        image_base64 straight to chat_stream(), but on the hybrid backend the
        reasoning model is text-only and DeepSeekBackend.chat_stream ignores
        the argument entirely. So on every typed turn carrying an image the
        frame reached neither model — never described by Groq/qwen, never seen
        by DeepSeek — while _build_reason_prompt was still told has_image=True.
        The model answered confidently about a picture nobody had looked at.

        That silently broke request_camera, whose entire purpose is delivering
        a frame: the client captured and uploaded one on request and the server
        binned it on arrival. Live ticks were unaffected (they run through
        _process_locked, which always had this block), which is exactly why it
        went unnoticed — watching worked, asking did not.

        Returns (scene_description, vision_prompt, goal_aware). The caller owns
        what happens next: _process_locked additionally runs its frame-buffer
        'scene unchanged' short-circuit, which must NOT apply to a typed turn —
        someone is waiting on an answer there and silence is never correct.
        """
        # Hand off the active goal to the vision stage instead of always
        # asking for a generic full-scene description. Qwen can't see
        # the task list — without this it wrote one paragraph of
        # everything, every tick, and DeepSeek (which never sees the
        # image at all — see the "image_base64=None" call below) had to
        # comb through that prose afterward to spot relevance itself.
        # DeepSeek already decided what to watch for when it called
        # request_live_search/update_task_list; this reads that
        # decision back out of the task-list state — rather than
        # spending a live per-frame reasoning call just to ask DeepSeek
        # what to look for, which would double the calls on every tick
        # for something it already told us via the task list.
        in_progress_items = [
            i for i in (tasklist.get_document() or {}).get("items", [])
            if i["status"] == "in_progress"
        ]
        in_progress = [i["content"] for i in in_progress_items]
        goal_aware = bool(in_progress)
        # The reasoning model's own brief to the vision stage, when it
        # wrote one (update_task_list's watch_for). Preferred over every
        # inference below, because those are guesses at intent from item
        # text — which is exactly what the "Find " prefix check downstream
        # exists to patch up. The model knows what it wants to know; this
        # is it saying so instead of us deducing it.
        #
        # Note this stays ONE vision call per frame: the brief is standing
        # state read back from the task list, not a per-tick round trip
        # asking the reasoning model what to look for (which would double
        # the calls on every frame — see the request_live_search notes).
        watch_briefs = [
            (i.get("watch_for") or "").strip()
            for i in in_progress_items
            if (i.get("watch_for") or "").strip()
        ]
        # An active "Find X" with no brief of its own still has to make it
        # into the request, or it would silently stop being searched for the
        # moment some *other* step got a watch_for. That's a live risk, not
        # a hypothetical: set_document deliberately keeps an in_progress
        # find-goal alive across full replaces that omit it, so one can
        # outlast the turn that created it and be the thing the user is
        # actually still waiting on.
        if watch_briefs:
            watch_briefs += [
                f"Is {i['content'][5:].strip()} visible in this frame? "
                f"If so, say where; if not, say what is in view instead."
                for i in in_progress_items
                if tasklist.is_find_goal_content(i["content"])
                and not (i.get("watch_for") or "").strip()
            ]
        # Only a "Find X" item (from request_live_search / start_find_task)
        # is a visual SEARCH — those, and only those, become object-
        # detection targets, with the "Find " prefix stripped to the bare
        # object. Other in_progress items are cooking STEPS ("Soak the
        # dal"), not things to locate; feeding a step to the detector as a
        # target produced nonsense ("find: Soak urad and rajma dal").
        find_targets = [c[5:].strip() for c in in_progress if c.lower().startswith("find ")]
        if watch_briefs:
            # Passed through close to verbatim. The wrapper bounds length
            # (the vision call shares the 8K TPM cap), forbids guessing —
            # an invented answer is worse than "can't tell", since the
            # reasoning model has no pixels to check it against — and holds
            # the stage boundary: this model reports, the reasoning model
            # judges. It cannot know the recipe or what "correct" looks
            # like, so an opinion from here is unanchored; measurements and
            # proportions, on the other hand, are exactly what a critique
            # downstream needs, so those are explicitly invited.
            #
            # Length scales with the brief rather than a flat cap: a
            # multi-part brief ("thickness, evenness, how much is left")
            # can't answer in two sentences, but a one-line brief
            # shouldn't be padded out to four.
            vision_prompt = (
                "You are the eyes of an assistant that cannot see this image. "
                "Answer ONLY the request(s) below, from what is actually "
                "visible in the frame. Give one short sentence per thing "
                "asked, four sentences maximum. If you cannot tell, say so "
                "plainly instead of guessing. Report what you observe — "
                "including rough measurements and proportions when asked — "
                "but do not judge whether it is being done well and do not "
                "give advice; that is the assistant's job, not yours.\n\n"
                + "\n".join(f"- {b}" for b in watch_briefs)
            )
        elif find_targets:
            goals_text = "; ".join(find_targets)
            # Object-detection directive, not a scene description. A strict,
            # tiny output format keeps this cheap and off the TPM cap;
            # anything beyond the format is wasted tokens the reasoning
            # stage must re-parse.
            vision_prompt = (
                f"OBJECT DETECTION. Target(s) to find: {goals_text}.\n"
                "Reply in EXACTLY one line, in one of these two forms and nothing else:\n"
                "FOUND: <a short phrase for where the target is in the frame>\n"
                "NOT FOUND: <a short phrase for what is in view instead>\n"
                "No preamble, no full-scene description, no colours/layout detail."
            )
        elif in_progress:
            # Active step(s) but nothing to visually search for — describe
            # the frame as it relates to the current step so the reasoning
            # stage sees relevant progress, without treating the step
            # itself as a detection target.
            steps_text = "; ".join(in_progress)
            vision_prompt = (
                f"The user is working on: {steps_text}. In 1-2 short "
                "sentences, describe only what in this image is relevant to "
                "that (progress, state, or a problem). No colours/layout "
                "detail, no advice."
            )
        else:
            # No active task — a one-line gist is all the reasoning stage
            # needs. Terse for the same TPM/latency reason as above.
            vision_prompt = (
                "In 1-2 short sentences, state only the main objects and "
                "what's happening in this image. No lists, no "
                "colours/textures/layout detail, no advice."
            )
        scene_description = await self.backend.vision(
            image_base64=image_base64,
            prompt=vision_prompt,
        )
        # Recorded separately from _record_debug_step (whose shape
        # assumes a VisionResponse) since this stage only returns a
        # plain string — without this, the debug UI showed nothing
        # between "frame received" and the reasoning call, making it
        # look like the reasoning backend (e.g. DeepSeek) had seen the
        # image itself rather than a separate vision model (Groq/qwen)
        # having described it first.
        debug_steps.append({
            "label": "vision" + (" (goal-aware)" if goal_aware else ""),
            "prompt_sent": vision_prompt,
            "has_image": True,
            "think": False,
            "tools_offered": [],
            "response_text": scene_description,
            "response_reasoning": "",
            "truncated": False,
            "tool_calls_raw": [],
            "model": getattr(self.backend, "vision_model", "unknown"),
            "provider": "groq",
        })
        return scene_description, vision_prompt, goal_aware

    def _available_tools(
        self, has_image: bool, is_live_frame: bool, is_camera_followup: bool = False,
    ) -> list:
        """Tools worth offering for this turn — excludes request_camera/
        request_live_search once there's already an image to look at (a
        live tick or an image-attached message). Shared by both the native
        tool-calling path (native_tools) and the prose tool-list built into
        the prompt for non-native backends, so the two can't drift apart —
        see CLAUDE.md's "second-opinion review" notes on this exact gap.

        is_camera_followup also suppresses them: this turn IS the response to a
        prior request_camera, so re-offering the tool just lets the model ask
        for a frame again instead of answering — the double-fire loop seen in
        real transcripts (Phase B kept coming back needs_camera=true because it
        arrived with no image attached and request_camera was still on offer).
        Suppressing on the followup breaks that loop regardless of whether the
        frame actually made it: with no image, the model answers with what it
        has (or says it couldn't see) rather than looping.
        """
        offer_camera = not has_image and not is_live_frame and not is_camera_followup
        camera_tool_names = {"request_camera", "request_live_search"}
        return [t for t in self.tools.list_tools() if t.name not in camera_tool_names or offer_camera]

    def _build_reason_prompt(
        self,
        prompt: str,
        scene: Optional[str],
        has_image: bool = False,
        think: bool = True,
        is_live_frame: bool = False,
        strip_task_list: bool = False,
        is_camera_followup: bool = False,
    ) -> str:
        """Build the prompt for the reasoning model."""
        # Persona doubles as a role brief. The single most-reported live-testing
        # failure was the model answering "what's next?" by pointing at the list
        # ("start with the in-progress item") instead of directing the user — it
        # only stopped once the user scolded it into understanding its job. That
        # job is stated up front here so it doesn't need re-teaching every
        # session: a hands-free voice guide whose output is heard, not read, and
        # whose job is to actively run the session and give the concrete next
        # action itself.
        persona = (
            "You are Chitragupt, a hands-free voice assistant for someone whose "
            "hands are busy — cooking, repairing something, shopping, working "
            "through any hands-on task. They are listening to "
            "you, not reading: every word is read aloud, so they cannot see any "
            "list, screen, or plan. Your job is to actively run the session for "
            "them — tell them the one thing to do right now in plain, concrete "
            "terms, then the next thing when they're ready, and keep track of "
            "everything in flight (steps, substitutions, things running in "
            "parallel, timers) "
            "so they never have to hold it in their head or ask you to check. "
            "Take initiative and give the actual instruction. Never answer by "
            "pointing them at a list, telling them to check what's next, or "
            "saying you've updated something and stopping there — figuring out "
            "and voicing the next step is your job, not theirs."
        )
        parts = [
            persona + (" You have tools to help you do this." if settings.TOOLS_ENABLED else "")
        ]

        if scene:
            parts.append(f"\n[Camera feed]\n{scene}")
        elif has_image:
            parts.append(
                "\n[Camera feed attached]\nAn image is attached below — look at "
                "it directly to answer, describing relevant details as needed."
            )

        # Running timers were previously invisible to the model — it started
        # them and then never heard about them again, so it couldn't answer
        # "how long left on the eggs?" and had no way to name one for
        # cancel_timer. This is pure arithmetic (timers.active_progress), so
        # it costs nothing per turn beyond the few tokens it renders to.
        active_timers = timers.active_progress()
        if active_timers:
            lines = []
            for t in active_timers:
                remaining = t["remaining_seconds"]
                mins, secs = divmod(remaining, 60)
                left = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
                lines.append(f"- '{t['label']}' — {left} left ({t['percent_done']}% done)")
            parts.append("\n[Timers]\n" + "\n".join(lines))
            parts.append(
                "These are running in the background right now. You'll be told "
                "automatically when one finishes, so don't wait on them or ask "
                "about them. If the user wants one stopped — they changed their "
                "mind, finished early, or are dropping the step — call "
                "cancel_timer with the label exactly as written above."
            )

        doc_summary = tasklist.render_summary(
            tasklist.get_document(), lean=is_live_frame, observations=not strip_task_list,
        )
        if doc_summary:
            parts.append(f"\n[Task list]\n{doc_summary}")
            parts.append(
                "Indented lines under an item are observations already logged "
                "against it — your memory of what's been seen so far. Check "
                "these before answering instead of assuming only the current "
                "frame/message is all you know."
            )
            if not is_live_frame:
                # The single most-reported failure in live testing: on a
                # "what's next?" turn the model answered by pointing AT the
                # list ("start with the in-progress item", "I've updated the
                # list") instead of speaking the actual step. That's useless
                # for a voice user who can't see the list — the reply is read
                # aloud. Force the concrete step, spoken, every time.
                parts.append(
                    "This list is read aloud — the user cannot see it. When they "
                    "ask what to do next (or what's happening now), do not refer to "
                    "the list itself. Never reply with a meta-answer like \"start "
                    "with the in-progress item\", \"check the list\", or \"I've "
                    "updated the list\". Instead, say the one current step out loud "
                    "in concrete, do-this-now terms — the actual action, not the "
                    "list mechanics. Give just that single current step, not the "
                    "whole plan."
                )

        parts.append(f"\n[User]\n{prompt}")

        tool_instruction = ""
        if settings.TOOLS_ENABLED:
            # request_camera/request_live_search only make sense when this
            # turn has no image to look at yet — a live-frame ping or an
            # image-attached message already has one, and offering either
            # tool there just invites the model to ask for something it
            # already has (or, for request_live_search, to re-start watching
            # that's already running).
            offer_camera = not has_image and not is_live_frame and not is_camera_followup
            tools = self._available_tools(has_image, is_live_frame, is_camera_followup)
            tool_list = "\n".join(
                f"- {t.name}({', '.join(t.parameters)}): {t.description}"
                for t in tools
            )
            # Native function-calling backends (Groq — see
            # VisionBackend.SUPPORTS_NATIVE_TOOLS) get tool calls through a
            # structured API field, not by hand-writing JSON into the
            # visible response — telling them to do both would just invite
            # a redundant/malformed text block alongside the real call.
            native = self.backend.SUPPORTS_NATIVE_TOOLS
            format_instruction = (
                ""
                if native
                else (
                    "\n\nYou have tools available. To call one, write this in your "
                    "visible response, not inside a <think> block (only the visible "
                    "response is checked for tool calls):\n"
                    '```tool\n{"name": "tool_name", "arguments": {"arg1": "value"}}\n```\n'
                )
            )
            tool_instruction = (
                format_instruction
                + f"Available tools:\n{tool_list}\n\n"
                "Tool calls execute instantly — you don't wait for a result or get "
                "a second turn to add more before the user sees your reply. Write "
                "your complete response — the actual answer, plan, or next step — "
                "in this SAME message as any tool call. Never stop after only "
                "announcing what you're about to do (e.g. 'Let me update the task "
                "list' or 'Sure, updating that now' with nothing else) — that reads "
                "as a dead end to the user, who then has to prompt you again just "
                "to get the guidance you already had. Say the placeholder and the "
                "substance together, or skip the placeholder and just say the "
                "substance.\n\n"
                "Tool-specific guidance:\n"
                "- start_timer: for a step that needs an actual wait (boiling, baking, "
                "marinating, steeping) — but ONLY once that step has genuinely STARTED. "
                "That means the user has told you they've begun it (\"the eggs are on\", "
                "\"I put it on the stove\", \"start the timer\") or you can clearly see it "
                "has started in the camera. If they are only planning or saying what they "
                "want to do next (\"I want to boil eggs\"), do NOT start a timer yet — "
                "acknowledge, and ask them to tell you when they've actually started it "
                "(or to say 'start the timer'). Never start one preemptively. Once "
                "started it runs in the background for free — don't wait for it or ask "
                "about it again yourself; completion is announced automatically. Then keep "
                "helping with whatever's next.\n"
                "- update_task_list: use whenever you're guiding a multi-step task (a "
                "recipe, a repair, a shopping trip, a project)."
                + (
                    " For a plain 'help me find X' with no other steps involved, use "
                    "request_live_search instead — it registers the goal for you."
                    if offer_camera else ""
                )
                + " Do this FIRST, before anything else, so later frames "
                "have a goal to check against. Always send the FULL item list, even items already "
                "completed — anything you leave out is dropped. Mark finished items "
                "'completed' rather than removing them, and use 'skipped' with a note for "
                "substitutions. Only call this tool when the list actually CHANGES — a new "
                "task, a step you're starting or finishing, or a substitution. If the user "
                "is only asking what to do next and nothing has changed, do NOT call it "
                "again — just answer from the [Task list] shown above. Reading it back is "
                "not a reason to call the tool. Set 'watch_for' on whichever item is "
                "in_progress to tell the camera what to check: you do not see the "
                "camera yourself — a separate vision model looks for you and reports "
                "back only what you asked for, so write it as a brief for someone "
                "looking on your behalf ('read the ingredient list and flag any "
                "gelatin', 'tell me when the onions are golden'). Without it that "
                "model only gets a generic description request and won't read labels "
                "or watch for your specific condition. Update it whenever the step "
                "changes. When you do reply, don't dump the whole "
                "plan, but always speak the single current step in concrete terms (the "
                "list is read aloud; the user can't see it). Each item MUST use the exact "
                "key 'content' for its text — not 'task' or 'label'."
                + (
                    ' Example:\n```tool\n{"name": "update_task_list", "arguments": '
                    '{"title": "Replace laptop battery", "items": [{"content": "Undo '
                    'the back panel screws", "status": "in_progress"}, {"content": "Unclip the old battery", '
                    '"status": "pending"}]}}\n```\n'
                    if not native else "\n"
                )
                + "- log_observation: call this on every relevant frame for an in-progress "
                "task-list item — e.g. what you currently see related to it — even on "
                "turns where you don't say anything to the user. This is your memory "
                "across frames; a later question like 'where is X' should be answered by "
                "checking these logged notes, not just the current frame.\n"
                + (
                    "- request_camera: no image is attached to this message. If answering "
                    "needs a single look at the current scene, call this instead of "
                    "guessing — do not describe or assume what's currently visible. This "
                    "includes when the user is trying to show you something but hasn't "
                    "attached an image yet (e.g. 'can you see it now', 'here you go') — "
                    "call request_camera to actually prompt them for one, rather than just "
                    "explaining how to attach a photo manually. Telling them how to use "
                    "the interface is only the right answer if they explicitly asked how "
                    "the interface works, not as a substitute for actually looking.\n"
                    "- request_live_search: use when the user wants you to help FIND a "
                    "specific object and one frame won't be enough (they'll need to move "
                    "the camera around while you keep checking). This starts continuous "
                    "watching scoped only to that target — don't use it for general "
                    "step-by-step help or anything else, that's not enabled through this tool.\n"
                    if offer_camera else ""
                )
            )
        # Broadened from is_live_frame-only: any turn that has an image AND an
        # active task list should log_observation, not just automated watch
        # ticks — a direct image upload or a request_camera follow-up is just
        # as much "camera + task info received" as a live tick is, and was
        # previously only covered by the much weaker generic tool-description
        # line above, not this forceful "always call" instruction.
        if has_image and doc_summary:
            parts.append(
                f"\nCheck the current frame against the [Task list] item(s) above and "
                f"their logged observations. Always call log_observation with what this "
                f"frame shows relevant to an in-progress item. If it's something to tell "
                f"the user about right now — progress, a problem, anything noteworthy — "
                f"pass alert=true on that call (this guarantees they're told even if you "
                f"don't write anything else this turn). Pass found=true ONLY on a 'Find X' "
                f"item whose target is now visible: that ends the search and switches the "
                f"camera off, so it is never the right flag for progress on a step."
            )
            if is_live_frame:
                # The silence protocol stays scoped to live ticks specifically —
                # a direct user turn (image upload, request_camera follow-up)
                # always got a real question and must always get a real answer.
                parts.append(
                    f"This is an automated watch tick, not a direct question. Only write "
                    f"a visible reply yourself if this frame changes something worth "
                    f"telling the user about (progress, a problem, the thing they're "
                    f"looking for). If nothing here is new or relevant, output the single "
                    f"token {SILENT_MARKER} as your entire visible reply — nothing before "
                    f"or after it. Do NOT narrate: writing a sentence like \"nothing "
                    f"relevant has changed\" or \"staying silent\" is itself wrong and "
                    f"leaks noise to the user. Either say the useful thing, or output only "
                    f"{SILENT_MARKER}."
                )
        thinking_instruction = (
            "\n\nThink step by step before responding."
            if think
            else "\n\nAnswer directly and concisely — no need for extended reasoning."
        )
        parts.append(
            thinking_instruction
            + tool_instruction
            + " Be concise, practical, and helpful in your final response."
            + " Respond in plain text only — no markdown (no **bold**, no headers, "
            "no bullet/numbered lists with * or -). This response may be read aloud "
            "by text-to-speech, so write it as plain spoken sentences."
        )

        return "\n".join(parts)

    def _extract_think_blocks(self, text: str) -> list[str]:
        """Extract <think>...</think> blocks from Qwen3 output."""
        return re.findall(r"<think>(.*?)</think>", text, re.DOTALL)

    def _remove_think_blocks(self, text: str) -> str:
        """Strip <think>...</think> blocks to get the visible response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _strip_tool_blocks(self, text: str) -> str:
        """Remove raw tool-call syntax, leaving only the model's own prose."""
        text = re.sub(r"```tool\s*\n?{.*?}\n?```", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _is_silent_live_reply(self, text: str) -> bool:
        """Whether a live-tick reply should be suppressed to silence.

        Covers three shapes, in order of confidence:
        1. an empty reply,
        2. SILENT_MARKER appearing anywhere (model emitted the token, maybe
           wrapped in stray prose the exact-match check would have missed),
        3. a reasoning model narrating its own silence instead of emitting the
           token at all — the observed DeepSeek/hybrid failure mode.

        Case 3 is gated on a length cap: a short reply that reads as a
        no-change declaration is noise, but a long, substantive update that
        happens to mention one of these phrases must still reach the user.
        Only ever called for is_live_frame turns — a direct user turn always
        gets a real answer regardless.
        """
        stripped = text.strip()
        if not stripped:
            return True
        if SILENT_MARKER in stripped.upper():
            return True
        if len(stripped) <= 300 and _SILENCE_NARRATION_RE.search(stripped):
            return True
        return False

    def _is_list_meta_nonanswer(self, text: str) -> bool:
        """Whether a direct-turn reply pointed the user AT the task list or its
        status ("start with the in-progress item", "I've updated the list")
        instead of voicing the concrete next step.

        Enforcement backstop for the top live-testing failure. Deliberately
        conservative — three conditions must all hold, so a real answer is
        never re-prompted:
        1. short (the bare non-answers say nothing actionable, so they run
           short; a reply that actually states a step + mentions the list in
           passing runs longer),
        2. uses list-mechanics vocabulary (_LIST_META_RE), and
        3. contains NO action imperative (_ACTION_CUE_RE) — if it does, it's
           voicing a step and has done its job regardless of any list mention.

        Only ever consulted on non-live direct turns; a live tick is allowed to
        be silent and a bare status note there is expected, not a failure.
        """
        stripped = text.strip()
        if not stripped or len(stripped) > 240:
            return False
        if _ACTION_CUE_RE.search(stripped):
            return False
        return bool(_LIST_META_RE.search(stripped))

    async def _voice_concrete_step(
        self, prompt: str, debug_steps: list[dict], label: str,
    ) -> Optional[str]:
        """One corrective call that forces the concrete next step out loud.

        Used by the enforcement backstop when a direct-turn reply was a
        list-meta non-answer. Builds a small self-contained prompt (just the
        current plan + a hard instruction) rather than replaying the full
        reasoning prompt, and passes no tools — this turn's only job is to
        speak the step, not to touch state again. Returns the corrected text,
        or None if the model somehow produced nothing usable (caller keeps the
        original reply in that case rather than blanking it).
        """
        doc = tasklist.get_document()
        summary = tasklist.render_summary(doc, lean=False, observations=False) if doc else ""
        correct_prompt = (
            "You are Chitragupt, a hands-free voice assistant guiding someone "
            "through a hands-on task. Your last "
            "reply pointed the user at the plan or its status instead of telling "
            "them what to do — but they can't see any list, it is read aloud to "
            "them. Here is the current plan for your reference only:\n\n"
            f"[Task list]\n{summary}\n\n"
            f"The user said: {prompt}\n\n"
            "Reply now in one or two short spoken sentences: the single concrete "
            "action to do right this moment — the actual physical step, in the "
            "terms of whatever they are doing (e.g. 'Heat some oil in a pan and "
            "add a teaspoon of cumin seeds', or 'Undo the two screws on the back "
            "panel and lift it straight off'). "
            "Do not mention the list, the steps, or their status. Do not say you "
            "have updated anything. Plain spoken text, no markdown."
        )
        correction = await self.backend.chat(
            image_base64=None, prompt=correct_prompt, think=False,
        )
        self._record_debug_step(debug_steps, label, correct_prompt, False, False, None, correction)
        corrected = self._strip_tool_blocks(correction.text).strip()
        return corrected or None

    def _run_structured_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls already parsed by a native-tool-calling backend
        (VisionResponse.tool_calls: [{"id", "name", "arguments"}, ...]) —
        no text-scanning involved, so there's no "wrong field name"/"missing
        arguments wrapper" failure mode to guard against here the way
        _execute_tool_calls has to for the regex path. Still catches
        TypeError for the rarer case of a required argument the model
        genuinely omitted despite the schema declaring it required.
        """
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            tool = self.tools.get(tool_name)
            if not tool:
                results.append({"tool": tool_name, "arguments": arguments, "result": f"Unknown tool: {tool_name}"})
                continue
            try:
                result = tool.fn(**arguments)
            except TypeError as e:
                result = f"Invalid arguments for {tool_name}: {e}"
            results.append({"tool": tool_name, "arguments": arguments, "result": result})
        return results

    async def _execute_tool_calls(self, text: str) -> list[dict]:
        """Find and execute any tool calls in the response text. Fallback
        path for backends without SUPPORTS_NATIVE_TOOLS (see
        _run_structured_tool_calls for the native path).

        Supports two formats:
          - ```tool { "name": "...", "arguments": {...} } ```
          - <tool>tool_name: arg</tool>
        """
        results = []

        # Format 1: JSON tool blocks ```tool { ... } ```
        json_pattern = r"```tool\s*\n?({.*?})\n?```"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                call = json.loads(match.strip())
            except json.JSONDecodeError as e:
                results.append({"tool": "unknown", "arguments": {}, "result": f"JSON parse error: {e}"})
                continue

            tool_name = call.get("name")
            # The model sometimes writes {"name": ..., "items": [...]}
            # directly instead of the documented {"name": ..., "arguments":
            # {"items": [...]}} — fall back to everything except "name" so
            # a missing wrapper doesn't turn into a hard TypeError below.
            arguments = call.get("arguments")
            if arguments is None:
                arguments = {k: v for k, v in call.items() if k != "name"}

            tool = self.tools.get(tool_name)
            if not tool:
                results.append({"tool": tool_name, "arguments": arguments, "result": f"Unknown tool: {tool_name}"})
                continue
            try:
                result = tool.fn(**arguments)
            except TypeError as e:
                # Missing/extra/misnamed arguments — surface it as a normal
                # tool result instead of crashing the whole turn.
                result = f"Invalid arguments for {tool_name}: {e}"
            results.append({"tool": tool_name, "arguments": arguments, "result": result})

        # Format 2: <tool>name: arg</tool> (Qwen3 ReAct format from CLAUDE.md)
        simple_pattern = r"<tool>(.*?):\s*(.*?)</tool>"
        matches = re.findall(simple_pattern, text, re.DOTALL)

        for tool_name, arg in matches:
            tool_name = tool_name.strip()
            arg = arg.strip()
            tool = self.tools.get(tool_name)
            if tool:
                result = tool.fn(arg)
                results.append({"tool": tool_name, "arguments": {"_": arg}, "result": result})
            else:
                results.append({"tool": tool_name, "arguments": {}, "result": f"Unknown tool: {tool_name}"})

        return results

    def reset_conversation(self):
        """Clear conversation memory, frame buffer, and any active task document."""
        self.memory.clear()
        self.frame_buffer.clear()
        tasklist.clear_document()

    async def check_timers(self) -> dict:
        """Public entry point for the /v1/timers/check poll — acquires the
        turn lock itself, since this is called independently of process()."""
        async with self._lock:
            return await self._check_timers_locked()

    async def _check_timers_locked(self) -> dict:
        """Fire completion calls for any due timers, return completions + free progress.

        Called on every client poll, and opportunistically from process() too
        so an active conversation surfaces a completion immediately instead of
        waiting on the next poll tick. Checking due-ness and computing
        progress is pure arithmetic (timers.due_unfired / timers.active_progress)
        — Groq is only ever called once per timer, right here, only when it's
        actually done. Routed through the same prompt-building and
        tool-execution path as a normal turn, so the completion can also
        update the task list (mark the step done) rather than just narrate it.

        Assumes self._lock is already held by the caller (process() or the
        public check_timers() wrapper) — must not be called concurrently
        with itself, since timers.mark_firing/mark_fired do read-modify-write
        on the same file.
        """
        for t in timers.due_unfired():
            timers.mark_firing(t["id"])
            timer_prompt = (
                f"[Timer completed]\nThe timer '{t['label']}' just finished after "
                f"{t['duration_seconds']} seconds.\nContext: {t['context'] or 'N/A'}\n"
                "Give a brief, practical next-step update for the user. If this step "
                "is tracked in the task list, update it to reflect that it's done."
            )
            reason_prompt = self._build_reason_prompt(
                prompt=timer_prompt, scene=None, has_image=False, think=False,
            )
            native_tools = (
                self.tools.to_openai_tools()
                if settings.TOOLS_ENABLED and self.backend.SUPPORTS_NATIVE_TOOLS
                else None
            )
            timer_debug_steps: list[dict] = []
            try:
                response = await self.backend.chat(
                    image_base64=None, prompt=reason_prompt, think=False, tools=native_tools,
                )
                self._record_debug_step(timer_debug_steps, "timer completion", reason_prompt, False, False, native_tools, response)
                clean_text = self._remove_think_blocks(response.text).strip()

                tool_results = []
                if settings.TOOLS_ENABLED:
                    if response.tool_calls:
                        tool_results = self._run_structured_tool_calls(response.tool_calls)
                    else:
                        tool_results = await self._execute_tool_calls(clean_text)
                    tool_results = [
                        r for r in tool_results
                        if not r["result"].startswith("Unknown tool:")
                        and not r["result"].startswith("JSON parse error:")
                        and not r["result"].startswith("Invalid arguments")
                    ]

                if tool_results and any(self.tools.get(r["tool"]).needs_followup for r in tool_results):
                    tool_context = "\n\n".join(
                        f"Tool '{r['tool']}' returned:\n{r['result']}" for r in tool_results
                    )
                    followup_prompt = (
                        f"I called tools while handling this timer completion. Results:\n\n"
                        f"{tool_context}\n\n{timer_prompt}\n"
                        "Please give the final brief update for the user."
                    )
                    final_response = await self.backend.chat(
                        image_base64=None,
                        prompt=followup_prompt,
                    )
                    self._record_debug_step(timer_debug_steps, "timer completion follow-up", followup_prompt, False, True, None, final_response)
                    message = final_response.text.strip()
                elif tool_results:
                    message = self._strip_tool_blocks(clean_text) or "\n".join(
                        r["result"] for r in tool_results
                    )
                else:
                    message = clean_text

                timers.mark_fired(t["id"], message, debug={"steps": timer_debug_steps, "tool_calls": tool_results})
            except Exception as e:
                # Leave it unfired — it'll be retried on the next poll instead
                # of blocking other timers' progress from being returned this tick.
                logger.error(f"Timer completion call failed for '{t['label']}': {e}")

        return {
            "completed": timers.pop_completions(),
            "active": timers.active_progress(),
        }
