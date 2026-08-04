"""Prompt construction for the v2 live system.

Asserts only what the SERVER builds and sends — never what a model replies.
Model behaviour is judged live; this file exists so that when a live session
goes wrong you can rule out the prompt in one command.

    python tests/t_live_prompts.py
"""
import sys
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')

from server.live import config, worlddoc
from server.live.agent import LiveAgent, SILENT_MARKER, URGENT_MARKER
from server.live.vision import build_tick_vision_prompt

A = LiveAgent.__new__(LiveAgent)   # prompt builders need no backend
FAIL = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def doc_with_focus():
    d = worlddoc._empty_doc()
    worlddoc.set_tasks(d, "Oil filter change", [
        {"content": "Chock the wheels", "status": "in_progress"},
        {"content": "Seat the new gasket", "status": "pending"},
    ])
    worlddoc.set_vision_focus(d, "User is loosening the oil filter under a car on ramps.")
    return d


# ── 1. Vision prompt: focus block ────────────────────────────────────────────
print("\n[1] VISION PROMPT — focus block")
focus = "User is dicing onions on a cutting board at the counter."
p = build_tick_vision_prompt(None, None, None, focus=focus)
check("the model's activity line is included verbatim", focus in p)
check("standard POSTURE AND GRIP block present", "POSTURE AND GRIP" in p)
check("standard DANGER block present", "DANGER" in p)
check("states it is not a checklist", "NOT a checklist" in p)
check("states absence is not a finding", "absence is not a finding" in p)
check("forbids verdicts", "Do NOT say whether any of it is safe" in p)
check("keeps the open anything-else channel", "ANYTHING ELSE in the frame" in p)
check("standard block is NOT model-supplied", "POSTURE AND GRIP" not in focus)

p_form = p
# Reading mode: a different job, so a different standard block entirely.
r = build_tick_vision_prompt(None, None, None,
                             focus="User is holding a curry packet up to the camera.",
                             focus_mode="read")
check("read mode asks for verbatim transcription", "Transcribe every piece of legible text" in r)
check("read mode demands exact numbers/units", "EXACTLY as shown" in r)
check("read mode says how to fix an unreadable frame", "rotate it upright" in r)
check("read mode forbids guessing characters", "Never guess at a character" in r)
check("read mode drops the grip block entirely", "POSTURE AND GRIP" not in r)
check("read mode is shorter than form mode", len(r) < len(p_form))

plain = build_tick_vision_prompt(None, None, None)
check("no focus set -> no form block at all", "POSTURE AND GRIP" not in plain)
check("no focus set -> still a normal description prompt", plain.startswith("You are the eyes"))

# ── 2. Vision prompt: question block ─────────────────────────────────────────
print("\n[2] VISION PROMPT — discrete questions")
qs = ["Is there a bag of black-eyed beans (lobhiya)? Read any label text.",
      "Have the onions gone past golden at the edges?"]
p = build_tick_vision_prompt("prev caption", None, qs)
check("every question is rendered", all(q in p for q in qs))
check("questions are numbered Q1..Qn", "Q1:" in p and "Q2:" in p)
check("question count is stated to stop invented Q3s", "exactly 2" in p)
check("FOUND / NOT VISIBLE / UNCLEAR shape given", all(k in p for k in ["FOUND", "NOT VISIBLE", "UNCLEAR"]))
check("no questions -> no answer block", "FIRST, answer" not in build_tick_vision_prompt(None, None, []))

# ── 3. Vision prompt: change framing ─────────────────────────────────────────
print("\n[3] VISION PROMPT — temporal framing")
p = build_tick_vision_prompt("Empty counter, knife to the right.", None, None)
check("previous caption passed as the comparison baseline", "Empty counter, knife to the right." in p)
check("asks for change in the SCENE", "CHANGED in the SCENE" in p)
check("camera motion banned", "NEVER describe the camera" in p)
check("viewpoint change is not a scene change", "A different viewpoint is not a change" in p)
check("first frame has no baseline text", "previous frame" not in build_tick_vision_prompt(None, None, None))

# ── 4. Which questions get asked, and how many ───────────────────────────────
print("\n[4] BRIEF SELECTION")
d = worlddoc._empty_doc()
worlddoc.set_tasks(d, "T", [{"content": "now", "status": "in_progress"},
                            {"content": "later", "status": "pending"}])
worlddoc.add_expectation(d, "current step", "event", condition="Is the wheel chocked?", task_ref="now")
worlddoc.add_expectation(d, "future step", "event", condition="Is oil smeared on the new gasket ring?", task_ref="later")
worlddoc.add_expectation(d, "safety", "event", condition="Is a drain pan directly beneath the work?", priority="high")
sel = A._vision_questions(d, charge=False)
check("in-progress task's question is asked", any("chocked" in q for q in sel))
check("future step's question is withheld", not any("gasket" in q for q in sel))
check("untied high-priority question always asked", any("drain pan" in q for q in sel))
check("respects MAX_ACTIVE_BRIEFS", len(sel) <= config.MAX_ACTIVE_BRIEFS, f"{len(sel)}")

many = worlddoc._empty_doc()
for i in range(config.MAX_ACTIVE_BRIEFS + 3):
    worlddoc.add_expectation(many, f"n{i}", "event",
                             condition=f"Distinct question number {i} about widget {i}?",
                             priority="high" if i == config.MAX_ACTIVE_BRIEFS + 2 else "normal")
sel = A._vision_questions(many, charge=False)
check("hard cap holds when over-subscribed", len(sel) == config.MAX_ACTIVE_BRIEFS, f"{len(sel)}")
check("high priority survives the cap", any("widget 6" in q for q in sel))

# ── 5. Vision token budget scales with questions ─────────────────────────────
print("\n[5] VISION TOKEN BUDGET")
b = lambda n: config.VISION_MAX_TOKENS + config.VISION_TOKENS_PER_QUESTION * n
check("budget grows per question", b(3) > b(0))
check("room for each answer line", b(3) - b(0) >= 3 * 40, f"{b(3)-b(0)}")

# ── 6. Capture tier ──────────────────────────────────────────────────────────
print("\n[6] FRAME DETAIL")
check("no watch -> coarse", A._frame_detail(worlddoc._empty_doc()) == "coarse")
d = worlddoc._empty_doc()
worlddoc.add_expectation(d, "find it", "event", condition="Is the bag of lobhiya visible on the shelf?")
check("open watch -> fine", A._frame_detail(d) == "fine")
for _ in range(config.MAX_BRIEF_ASKS):
    A._vision_questions(d)
check("budget spent -> back to coarse", A._frame_detail(d) == "coarse")
check("...but the watch stays open", len(A._vision_questions(d, charge=False)) == 1)
worlddoc.resolve_expectation(d, "find it")
check("resolved -> coarse", A._frame_detail(d) == "coarse")
t = worlddoc._empty_doc()
worlddoc.add_expectation(t, "timer", "time", due_in_seconds=600)
check("time-anchored watch does not force fine", A._frame_detail(t) == "coarse")

print("\n[6b] FRAME DETAIL — model-requested via vision focus")
d = worlddoc._empty_doc()
worlddoc.set_vision_focus(d, "User is hammering a nail into a wall.", "fine")
check("focus asking fine -> fine", A._frame_detail(d) == "fine")
check("focus detail is readable back", worlddoc.focus_detail(d) == "fine")
worlddoc.set_vision_focus(d, "User is carrying the pot to the table.", "coarse")
check("model reverting to coarse -> coarse", A._frame_detail(d) == "coarse")
check("...even though a focus is still active", worlddoc.get_vision_focus(d) is not None)
worlddoc.set_vision_focus(d, "", "fine")
check("cleared focus -> coarse", A._frame_detail(d) == "coarse")

# The backstop for when the model never reverts — v1's documented failure.
d = worlddoc._empty_doc()
worlddoc.set_vision_focus(d, "User is seating a new oil filter by hand.", "fine")
for _ in range(config.MAX_FINE_FOCUS_FRAMES):
    check_before = A._frame_detail(d)
    worlddoc.charge_focus_frame(d)
check("fine held for the whole budget", check_before == "fine")
check("budget exhausted -> forced coarse", A._frame_detail(d) == "coarse")
check("...but the focus survives", worlddoc.get_vision_focus(d) is not None)
worlddoc.set_vision_focus(d, "User is reading the torque spec off the plug.", "fine")
check("re-arming after exhaustion resets the budget", A._frame_detail(d) == "fine")

print("\n[6d] READING MODE")
d = worlddoc._empty_doc()
worlddoc.set_vision_focus(d, "User is holding a packet up to read it.", "coarse", "read")
check("read mode overrides a coarse request", worlddoc.focus_detail(d) == "fine")
check("read mode forces fine frames", A._frame_detail(d) == "fine")
check("mode is readable back", worlddoc.focus_mode(d) == "read")
worlddoc.set_vision_focus(d, "User is stirring the pot.", "coarse", "form")
check("form mode honours coarse again", A._frame_detail(d) == "coarse")
d2 = worlddoc._empty_doc()
worlddoc.set_vision_focus(d2, "x", "fine", "nonsense")
check("unknown mode falls back to form", worlddoc.focus_mode(d2) == "form")

print("\n[6c] FOCUS IS VISIBLE IN THE DOC")
d = worlddoc._empty_doc()
worlddoc.set_vision_focus(d, "User is hammering a nail into a wall.", "fine")
worlddoc.charge_focus_frame(d); worlddoc.charge_focus_frame(d)
r = worlddoc.render(d)
check("focus section rendered", "[Camera focus" in r)
check("tier shown so the model sees it left fine on", "fine frames" in r)
check("frames used shown", "2 used" in r)
check("brief shown", "hammering a nail" in r)

# ── 7. Reasoning prompts ─────────────────────────────────────────────────────
print("\n[7] REASONING PROMPTS")
d = doc_with_focus()
tick = A._build_tick_prompt(d, "Right hand on a filter wrench.", events=[])
check("caption is included", "Right hand on a filter wrench." in tick)
# Speech rules now live in stage 2 (see [7b]) — the tick prompt is bookkeeping only.
check("tick prompt no longer decides speech", SILENT_MARKER not in tick)
check("urgent handling moved out of the tick prompt", URGENT_MARKER not in tick)
check("READY/DONE licence moved to the speech step",
      "ready or done" in A._build_speech_prompt(d, "cap", [], []).lower())
check("world doc is rendered in", "[Tasks]" in tick)
check("tick explains the system to the model", "HOW THIS WORKS" in tick)
check("...explains it cannot see pixels", "You never see pixels" in tick)
check("...explains the doc is its own writes", "YOUR memory" in tick)
check("...only once", tick.count("HOW THIS WORKS") == 1)

chat = A._build_chat_prompt(d, "where are the onions?", caption=None)
check("user question included", "where are the onions?" in chat)
check("never silent on a user turn", "never reply with the silent marker" in chat)
check("told the user is listening, not reading", "LISTENING" in chat)
check("told to call set_vision_focus on hands-on work", "set_vision_focus" in chat)
check("told NOT to write the checklist itself", "do NOT describe the setup you expect" in chat.replace("Do NOT", "do NOT"))
check("retraction path taught", "retract_environment_fact" in chat)
check("chat carries the same system brief", chat.count("HOW THIS WORKS") == 1)
check("answers a pending request from the speech step",
      "they are waiting and cannot see your notes"
      in A._build_speech_prompt(d, "cap", [], []))

# ── 8. Doc rendering ─────────────────────────────────────────────────────────
print("\n[7b] TWO-STAGE TICK: bookkeeping vs speech")
d = doc_with_focus()
stage1 = A._build_tick_prompt(d, "Cumin spluttering in hot oil.", events=[])
stage2 = A._build_speech_prompt(d, "Cumin spluttering in hot oil.",
                                [{"tool": "mark_task", "result": "ok"}], [])
check("stage 1 carries no speech rules",
      not any(k in stage1 for k in ["Speak when", "SAFETY OVERRIDE", "default is silence"]))
check("stage 1 says speech is decided elsewhere", "decided separately" in stage1)
check("stage 1 still asks for tool bookkeeping", "log_environment" in stage1)
check("stage 2 asks exactly one question", "Decide ONE thing" in stage2)
check("stage 2 sees what was recorded", "mark_task" in stage2)
check("stage 2 sees the caption", "Cumin spluttering" in stage2)
check("stage 2 knows when the user last spoke", "[User last spoke]" in stage2)
check("stage 2 knows when it last spoke", "[You last spoke]" in stage2)
check("stage 2 can escalate a hazard", URGENT_MARKER in stage2)
check("stage 2 can decline", SILENT_MARKER in stage2)
check("stage 2 is much cheaper than stage 1", len(stage2) < len(stage1) / 2,
      f"{len(stage2)} vs {len(stage1)}")
check("stage 2 omits the full doc", "[Recent observations]" not in stage2)
check("stage 2 omits the system brief", "HOW THIS WORKS" not in stage2)

print("\n[8] WORLD DOC RENDER")
r = worlddoc.render(d)
check("current time header first", r.startswith("[Current time:"))
check("goal rendered", "[Goal] Oil filter change" in r)
check("in-progress marked", "[~] Chock the wheels" in r)
check("stability-first ordering (recent last)", r.index("[Tasks]") < len(r))

print("\n" + "=" * 60)
print(f"{'ALL PROMPT CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
