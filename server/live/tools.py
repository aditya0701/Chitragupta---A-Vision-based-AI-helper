"""Tool registry for the live system.

Reuses the Tool/ToolRegistry classes and the info-gathering tool functions
(web_search, fetch_page, calculate) from server.agent unchanged. The
doc-mutating tools are new: they close over a `get_doc` callable supplied by
LiveAgent, mutating the in-memory doc for the current turn — the agent owns
loading it at turn start and persisting it at turn end, so a turn's tool
calls and the agent's own writes can never interleave on disk.

Deliberately absent vs. the old system:
  start_timer        subsumed by a time-anchored expectation — same wall-
                     clock arithmetic, plus a resolution path timers never
                     had (satisfied silently if confirmed before deadline).
  request_camera /   the live UI owns the camera; chat turns attach the
  request_live_search  current frame client-side whenever the stream is on.
"""

from __future__ import annotations

from typing import Callable

from ..agent import (
    Tool,
    ToolRegistry,
    tool_calculate,
    tool_fetch_page,
    tool_web_search,
)
from . import worlddoc


def build_live_tools(get_doc: Callable[[], dict]) -> ToolRegistry:
    registry = ToolRegistry()

    def _update_tasks(title: str, items: list) -> str:
        return worlddoc.set_tasks(get_doc(), title, items)

    def _set_expectation(
        description: str,
        anchor: str,
        due_in_seconds: float = None,
        condition: str = None,
        priority: str = "normal",
        task: str = None,
    ) -> str:
        return worlddoc.add_expectation(
            get_doc(), description, anchor,
            due_in_seconds=due_in_seconds, condition=condition,
            priority=priority, task_ref=task,
        )

    def _resolve_expectation(ref: str, outcome: str = "satisfied", note: str = "") -> str:
        return worlddoc.resolve_expectation(get_doc(), ref, outcome, note)

    def _log_environment(fact: str) -> str:
        return worlddoc.add_environment_fact(get_doc(), fact)

    def _retract_environment_fact(fact_match: str, correction: str = "") -> str:
        return worlddoc.retract_environment_fact(get_doc(), fact_match, correction)

    def _mark_task(task: str, status: str, note: str = "") -> str:
        doc = get_doc()
        match = worlddoc.find_task(doc, task)
        if not match:
            return f"No task matching '{task}' — check the [Tasks] content exactly."
        if status not in worlddoc.VALID_TASK_STATUSES:
            return f"status must be one of {sorted(worlddoc.VALID_TASK_STATUSES)}."
        match["status"] = status
        if note:
            match["note"] = note
        worlddoc.touch_task(doc, match["id"])
        return f"Task '{match['content']}' marked {status}."

    registry.register(Tool(
        name="update_tasks",
        description=(
            "Create or fully replace the plan — the persistent record of what needs doing. "
            "Send the FULL list every time (items you omit are dropped). Statuses: pending, "
            "in_progress, completed (keep in list), skipped (say why in note). For changing "
            "ONE item's status, prefer mark_task instead of resending everything."
        ),
        fn=_update_tasks,
        parameters={
            "title": {"type": "string", "description": "Name of the overall goal, e.g. 'Chicken Biryani'", "required": True},
            "items": {
                "type": "array",
                "description": "Full list of task items.",
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The item's text — always this exact key, never 'task' or 'label'"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "skipped"]},
                        "note": {"type": "string", "description": "Optional, e.g. reason for a substitution"},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="mark_task",
        description="Change one task's status without resending the whole list.",
        fn=_mark_task,
        parameters={
            "task": {"type": "string", "description": "Exact task content or id", "required": True},
            "status": {"type": "string", "description": "pending | in_progress | completed | skipped", "required": True},
            "note": {"type": "string", "description": "Optional note", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="set_expectation",
        description=(
            "Register something that SHOULD happen so the system can notice if it doesn't. "
            "Two kinds: anchor='time' with due_in_seconds (e.g. 'rice should be started', due "
            "in 600s — if not confirmed by then, the user gets reminded automatically, at zero "
            "cost while waiting). anchor='event' with condition — this is ALSO how you aim the "
            "camera. You cannot see; a separate vision model looks for you, and `condition` is "
            "put to it verbatim as a question on every frame until you resolve it. So phrase it "
            "as an answerable OBSERVATION question, not a statement and not a judgement call: "
            "'Is there a bag of black-eyed beans (lobhiya)? Small white beans with a black "
            "spot — read any label text.' rather than 'black eyed beans are visible'. Ask for "
            "what is legible or visibly distinctive; you do the deciding. Use this whenever the "
            "user is looking for something specific — a generic caption will not find it. "
            "ALSO use it for how the work is being DONE, on any hands-on task: SAFETY at "
            "priority='high' (hand in the path of a blade, handle past the edge, tool about to "
            "slip) and FORM at priority='normal' (grip on a knife or spanner, whether a jaw sits "
            "square on the flats, whether a pan is crowded, whether a driver is seated square). "
            "Form watches matter even when nothing is dangerous — they decide whether the job "
            "comes out well, and the user cannot watch their own hands. Ask for the physical "
            "detail ('is the jaw square on the flats or angled across the corners?'), never for "
            "a judgement ('is this correct?') — you are the one who judges. "
            "Set priority='high' "
            "only for things worth interrupting the user over; 'low' waits for a natural moment. "
            "When you help plan anything with timings, set the expectations in the same turn — "
            "don't wait to be asked."
        ),
        fn=_set_expectation,
        parameters={
            "description": {"type": "string", "description": "What should happen, phrased so it's useful when read back later", "required": True},
            "anchor": {"type": "string", "enum": ["time", "event"], "required": True},
            "due_in_seconds": {"type": "number", "description": "time-anchored only: seconds from now until this is overdue", "required": False},
            "condition": {"type": "string", "description": "event-anchored only: what visible situation makes this fire", "required": False},
            "priority": {"type": "string", "enum": ["high", "normal", "low"], "required": False},
            "task": {"type": "string", "description": "Optional: task content/id this belongs to", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="resolve_expectation",
        description=(
            "Close an open expectation: outcome='satisfied' when you've confirmed it happened "
            "(from the frame or the user saying so), 'cancelled' when it no longer applies. "
            "Resolve satisfied expectations silently as you notice them — no need to announce it."
        ),
        fn=_resolve_expectation,
        parameters={
            "ref": {"type": "string", "description": "Expectation id or exact description", "required": True},
            "outcome": {"type": "string", "enum": ["satisfied", "cancelled"], "required": False},
            "note": {"type": "string", "description": "Optional context", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="set_vision_focus",
        description=(
            "Tell the camera WHAT THE USER IS DOING right now. Call this on any hands-on "
            "task, and update it as the work moves from step to step. "
            "ONE OR TWO SENTENCES, plain description of the activity — 'User is loosening "
            "the oil filter under a car that is up on ramps.' or 'User is dicing onions "
            "on a board at the counter.' That is all. "
            "Do NOT list things for it to check, and do NOT describe the setup you expect "
            "— no 'whether a drain pan is underneath', no 'whether a chock is behind the "
            "wheel'. You cannot see their kitchen or their garage, so any arrangement you "
            "imagine will be wrong for someone whose setup is merely different, and the "
            "camera will report your guesses back as problems that do not exist. "
            "The instructions for reporting grip, posture and danger are already attached "
            "automatically to every frame — you do not need to write them and you should "
            "not try. Just say what the activity is, and keep it current. "
            "Send an empty brief when the hands-on work is done. Keep set_expectation for "
            "discrete things that either are or aren't true yet. "
            "ALSO set `detail`. Use detail='fine' whenever the step turns on seeing SMALL "
            "things — where fingers sit relative to a blade, whether a socket is square on "
            "a nut, threads, label text, a gauge reading. Fine frames are 1024px and cost "
            "about 40% more, so use detail='coarse' when only gross movement matters — "
            "stirring, carrying, walking to a shelf, waiting for something to boil. "
            "SET mode='read' WHENEVER YOU NEED TO KNOW WHAT SOMETHING SAYS — a packet's "
            "cooking instructions, a bottle's ingredients, a torque figure, a dial, a "
            "screen, a model number. In read mode the camera transcribes the text verbatim "
            "and tells you what is illegible and how to fix it (rotate, closer, less "
            "glare), instead of describing grip and posture — which is useless when what "
            "you actually need is the words. Do not try to get a label read with a form "
            "brief; ask for read mode and you will get the text back. Switch to mode='form' "
            "once you have what you needed. "
            "IMPORTANT: when the thing you needed fine detail for is done — the step "
            "finished, the part is seated, you got your answer, the user moved on — call "
            "this again with detail='coarse' (or an empty brief if the hands-on work is "
            "over entirely). Do not leave it on fine out of habit; it is billed on every "
            "single frame until you change it."
        ),
        fn=lambda brief="", detail="fine", mode="form": worlddoc.set_vision_focus(
            get_doc(), brief, detail, mode),
        parameters={
            "brief": {"type": "string", "description": "One or two sentences: what the user is physically doing right now. Not a checklist. Empty string clears it.", "required": True},
            "detail": {"type": "string", "enum": ["fine", "coarse"], "description": "fine = 1024px, for seeing grip/threads/labels/small detail. coarse = 640px, for gross movement. Revert to coarse when the close work is done.", "required": False},
            "mode": {"type": "string", "enum": ["form", "read"], "description": "form = watch how the work is being done (grip, posture, danger). read = TRANSCRIBE text in frame verbatim — labels, instructions, gauges, screens. Forces fine frames.", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="log_environment",
        description=(
            "Record a durable fact about the physical environment worth remembering beyond this "
            "moment — where things are, what's available, what state something was left in. "
            "E.g. 'red chili powder is on the top shelf, left side'. Use the same place-words "
            "consistently across the session ('top shelf' stays 'top shelf') so facts stay "
            "matchable. This is your long-term spatial memory — log locations whenever you spot "
            "something the user might look for later, even if it's not relevant right now."
        ),
        fn=_log_environment,
        parameters={
            "fact": {"type": "string", "description": "One short, specific, durable fact", "required": True},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="retract_environment_fact",
        description=(
            "Undo a fact you logged that turns out to be wrong. ALWAYS call this the "
            "moment the user corrects something you recorded or claimed — 'no, those "
            "aren't lobhiya', 'that's not the toor dal', 'wrong shelf'. Pass a distinctive "
            "fragment of the wrong fact as fact_match, and what is actually true as "
            "correction. Do not just log a new fact on top: the old one keeps riding "
            "along in every prompt and you will repeat the mistake. If you are no longer "
            "sure what the item is, say so in the correction ('the bag on the pantry "
            "shelf is NOT black-eyed beans; contents unconfirmed') rather than guessing "
            "again — an honest unknown is worth more than a second wrong label."
        ),
        fn=_retract_environment_fact,
        parameters={
            "fact_match": {"type": "string", "description": "Distinctive text from the wrong fact, as it appears in [Known environment facts]", "required": True},
            "correction": {"type": "string", "description": "What is actually true — recorded in its place. Omit only if nothing is known.", "required": False},
        },
        needs_followup=False,
    ))

    registry.register(Tool(
        name="web_search",
        description="Search the web for information (e.g. a recipe's real timings before planning).",
        fn=tool_web_search,
        parameters={"query": {"type": "string", "description": "Search query", "required": True}},
    ))
    registry.register(Tool(
        name="fetch_page",
        description="Fetch a web page by URL and return its visible text content.",
        fn=tool_fetch_page,
        parameters={"url": {"type": "string", "description": "The URL to fetch", "required": True}},
    ))
    registry.register(Tool(
        name="calculate",
        description="Evaluate a mathematical expression.",
        fn=tool_calculate,
        parameters={"expression": {"type": "string", "description": "Math expression", "required": True}},
    ))

    return registry
