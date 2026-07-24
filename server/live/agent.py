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
from typing import Optional

from ..agent import ConversationMemory
from ..backends import VisionBackend, should_think
from . import compaction, triggers, worlddoc
from .tools import build_live_tools
from .vision import build_tick_vision_prompt

logger = logging.getLogger("chitragupt.live")

SILENT_MARKER = "[SILENT]"

PERSONA = (
    "You are Chitragupta, a live first-person assistant watching through the "
    "user's camera — an attentive record keeper. You track everything in a "
    "persistent world document (shown below) so the user never has to repeat "
    "themselves. Be brief and concrete when you speak."
)


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

    async def _reason(self, prompt: str, think: bool, history: Optional[list[dict]] = None):
        """One reasoning call + tool execution + at most one follow-up call
        (only when a tool surfaced new information), + one truncation retry."""
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

        tool_results = self._run_tool_calls(response.tool_calls)
        text = (response.text or "").strip()

        followup_results = [r for r in tool_results if r.get("needs_followup")]
        if followup_results:
            results_text = "\n\n".join(
                f"Result of {r['tool']}:\n{r['result']}" for r in followup_results
            )
            followup_prompt = (
                f"{prompt}\n\n[You called tools; here are the results — use them to give "
                f"your final answer now, without calling those tools again]\n{results_text}"
            )
            response2 = await self.backend.chat(
                image_base64=None, prompt=followup_prompt,
                conversation_history=history, think=False,
                tools=self._native_tools(),
            )
            extra_results = self._run_tool_calls(response2.tool_calls)
            tool_results.extend(extra_results)
            if (response2.text or "").strip():
                text = response2.text.strip()

        for r in tool_results:
            r.pop("needs_followup", None)
        return text, tool_results, response

    # ── Tick ─────────────────────────────────────────────────────────────────

    def _goal_hint(self, doc: dict) -> str:
        """Short goal text for the vision stage — goal-conditioned detail."""
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        parts += [t["content"] for t in doc["tasks"] if t["status"] == "in_progress"]
        parts += [f"watch for: {e['condition']}"
                  for e in worlddoc.open_expectations(doc) if e["anchor"] == "event"]
        return "\n".join(f"- {p}" for p in parts)

    def _build_tick_prompt(self, doc: dict, caption: str, events: list[dict]) -> str:
        lines = [
            PERSONA, "",
            worlddoc.render(doc), "",
            f"[New camera observation, {worlddoc.fmt_ts(doc['recent'][-1]['ts'])}]",
            caption, "",
            "This is an automatic camera tick, NOT a user message. The user is busy; "
            "your default is silence.",
            "",
            "Housekeeping (do silently via tools, this is most of your job):",
            "- If the frame confirms an open expectation happened, call resolve_expectation.",
            "- Check every event-anchored expectation's condition against this frame; if one "
            "fires (the condition is now true), speak up about it.",
            "- If the frame shows where something is kept, call log_environment.",
            "- If a task visibly finished or started, call mark_task.",
            "",
            f"Then: if nothing needs saying to the user, your entire visible reply must be "
            f"exactly {SILENT_MARKER} and nothing else. Speak only if: an event-anchored "
            "expectation fired, something genuinely new and important for the active goal "
            "happened, the user is about to make a mistake, or a trigger event below asks "
            "you to.",
        ]
        if events:
            lines += ["", "[Trigger events — these fired by arithmetic while you were away; "
                          "address them in your reply]"]
            lines += [f"- {e['text']}" for e in events]
        return "\n".join(lines)

    async def tick(self, image_base64: str) -> dict:
        async with self._lock:
            doc = worlddoc.load()
            self._doc = doc
            try:
                prev = worlddoc.last_caption(doc)
                vision_prompt = build_tick_vision_prompt(prev, self._goal_hint(doc) or None)
                caption = await self.backend.vision(image_base64, vision_prompt)

                batch = worlddoc.add_recent(doc, caption)
                if batch:
                    await compaction.compact(self.backend, doc, batch)

                events = triggers.check(doc)
                prompt = self._build_tick_prompt(doc, caption, events)
                text, tool_results, response = await self._reason(prompt, think=False)

                if text.upper() == SILENT_MARKER or not text:
                    text = ""
                else:
                    # Politeness budget: trigger-driven and high-priority speech
                    # always passes; spontaneous commentary waits out the gap.
                    important = bool(events) or any(
                        r["tool"] == "resolve_expectation" for r in tool_results
                    )
                    if not important and not triggers.may_speak_unprompted(doc):
                        logger.info("Politeness gate suppressed unprompted tick speech: %r", text[:80])
                        text = ""
                if text:
                    triggers.mark_spoke(doc)

                worlddoc.save(doc)
                return {
                    "text": text or None,
                    "caption": caption,
                    "triggers": [e["text"] for e in events],
                    "tool_calls": tool_results,
                    "model": response.model,
                    "provider": response.provider,
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
                    vision_prompt = build_tick_vision_prompt(
                        worlddoc.last_caption(doc), self._goal_hint(doc) or None)
                    caption = await self.backend.vision(image_base64, vision_prompt)
                    batch = worlddoc.add_recent(doc, caption)
                    if batch:
                        await compaction.compact(self.backend, doc, batch)

                built = self._build_chat_prompt(doc, prompt, caption)
                text, tool_results, response = await self._reason(
                    built, think=should_think(prompt),
                    history=self.memory.get_history(),
                )
                if not text:
                    text = "(no reply — something went wrong, try again)"

                self.memory.add("user", prompt)
                self.memory.add("assistant", text)
                triggers.mark_spoke(doc)  # answering counts — resets the nag clock too
                worlddoc.save(doc)
                return {
                    "text": text,
                    "caption": caption,
                    "tool_calls": tool_results,
                    "model": response.model,
                    "provider": response.provider,
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
