"""The find list, end to end through the real tick() and poll().

Replays the 2026-08-10 failure: the camera saw the chicken, the caption said
so in plain English, and the user was told nothing for 65 seconds because the
speech stage was free to decide it would be narrating.

The property under test is that a find is no longer a judgement call. Once the
camera answers FOUND, the user gets told — over a hostile politeness gate, and
even when the speech model itself declines.

    python tests/t_live_wanted.py
"""
import asyncio, pathlib, sys, tempfile, time

sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')

from server.live import worlddoc
worlddoc.DOC_FILE = pathlib.Path(tempfile.mkdtemp()) / "w.json"
from server.backends import VisionResponse
from server.live.agent import LiveAgent
from server.live import config

FAIL = []


def check(label, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"\n          {extra}" if extra and not cond else ""))
    if not cond:
        FAIL.append(label)


# The real caption from 11:39:42, with the find-list answer lines the camera
# now produces. Everything below the answers is verbatim from the session.
FOUND_CAPTION = (
    "chicken packet: FOUND - red vacuum-sealed pack on the second shelf of the fridge, right side\n"
    "onions: NOT VISIBLE - a white tub with a yellow label sits beside the pack\n"
    "\n"
    "View is inside the refrigerator, focused on the second shelf down. The user's "
    "right hand reaches in to grasp the edge of the chicken package."
)

QUIET_CAPTION = (
    "chicken packet: NOT VISIBLE - the counter holds a black pot and a shaker\n"
    "onions: NOT VISIBLE - no basket or bowl in this part of the frame\n"
    "\n"
    "View centers on the countertop adjacent to the sink. Nothing has changed."
)


class FakeBackend:
    """`speech` decides what the announce/speech call returns, so a test can
    make the model refuse and check that the user is told anyway."""
    SPLIT_VISION_REASONING = True
    SUPPORTS_NATIVE_TOOLS = True

    def __init__(self, caption, speech="Found it."):
        self.caption = caption
        self.speech = speech
        self.prompts = []

    async def vision(self, image_base64, prompt="", max_tokens=160):
        self.prompts.append(("vision", prompt))
        return self.caption

    async def chat(self, image_base64=None, prompt="", conversation_history=None,
                   think=True, tools=None):
        kind = "announce" if "[Just seen]" in prompt else "reason"
        self.prompts.append((kind, prompt))
        text = self.speech if kind == "announce" else "."
        return VisionResponse(text=text, model="fake", provider="fake",
                              truncated=False, tool_calls=[])


def seed(hostile_gate=False, announced=False):
    """A doc mid-search: the user asked 2 minutes ago, nothing found yet."""
    worlddoc.clear()
    doc = worlddoc._empty_doc()
    doc["title"] = "Chicken Curry"
    worlddoc.add_wanted(doc, ["chicken packet", "onions"],
                        because="I need your help in finding the chicken and onion")
    if announced:
        for w in doc["wanted"]:
            w["status"], w["where"], w["announced"] = "found", "somewhere", True
            w["found_ts"] = time.time()
    if hostile_gate:
        # Spoke one second ago: MIN_UNPROMPTED_GAP_S would normally gag this
        # tick completely, and the follow-up window is deliberately shut too so
        # nothing but the find itself can be what lets the words through.
        doc["last_spoken_ts"] = time.time() - 1
        doc["last_user_turn_ts"] = time.time() - (config.FOLLOWUP_WINDOW_S + 60)
    worlddoc.save(doc)
    return doc


async def main():
    print("\n1. the replay — camera answers FOUND, user is told on the same tick")
    seed()
    be = FakeBackend(FOUND_CAPTION,
                     speech="The chicken's in the fridge, second shelf on the right.")
    out = await LiveAgent(backend=be).tick("img")
    check("tick speaks", bool(out["text"]), repr(out["text"]))
    check("says where", "second shelf" in (out["text"] or ""))
    doc = worlddoc.load()
    check("chicken marked found", worlddoc.find_wanted(doc, "chicken packet")["status"] == "found")
    check("and marked told", worlddoc.find_wanted(doc, "chicken packet")["announced"] is True)
    check("onions still open", worlddoc.find_wanted(doc, "onions")["status"] == "open")
    check("location promoted to a durable fact",
          any("second shelf" in f["fact"] for f in doc["environment"]))
    ann = [p for k, p in be.prompts if k == "announce"]
    check("announce prompt used, not the speech prompt", len(ann) == 1)
    check("SILENT is never offered on it", "[SILENT]" not in (ann[0] if ann else "[SILENT]"))

    print("\n2. the camera's find beats a hostile politeness gate")
    seed(hostile_gate=True)
    be = FakeBackend(FOUND_CAPTION, speech="Chicken's on the second fridge shelf.")
    out = await LiveAgent(backend=be).tick("img")
    check("spoken 1s after the last utterance, outside the follow-up window",
          bool(out["text"]), repr(out["text"]))

    print("\n3. the model refuses — the server says it anyway")
    seed(hostile_gate=True)
    be = FakeBackend(FOUND_CAPTION, speech="[SILENT]")
    out = await LiveAgent(backend=be).tick("img")
    check("user is still told", bool(out["text"]), repr(out["text"]))
    check("fallback names the item", "chicken packet" in (out["text"] or "").lower())
    check("fallback names the place", "second shelf" in (out["text"] or ""))

    print("\n4. an empty reply is treated the same way")
    seed(hostile_gate=True)
    out = await LiveAgent(backend=FakeBackend(FOUND_CAPTION, speech="   ")).tick("img")
    check("user is still told", bool(out["text"]), repr(out["text"]))

    print("\n5. told once, never repeated")
    seed(announced=True)
    out = await LiveAgent(backend=FakeBackend(QUIET_CAPTION)).tick("img")
    check("silent tick", not out["text"], repr(out["text"]))

    print("\n6. nothing found — the ordinary silent path is untouched")
    seed(hostile_gate=True)
    out = await LiveAgent(backend=FakeBackend(QUIET_CAPTION)).tick("img")
    check("stays silent", not out["text"], repr(out["text"]))
    doc = worlddoc.load()
    check("misses counted", worlddoc.find_wanted(doc, "onions")["misses"] == 1)

    print("\n7. found, then the camera dies — poll() still delivers it")
    seed(hostile_gate=True)
    doc = worlddoc.load()
    worlddoc.fold_wanted(doc, FOUND_CAPTION)
    worlddoc.save(doc)
    out = await LiveAgent(backend=FakeBackend("", speech="[SILENT]")).poll()
    check("poll announces", bool(out["message"]), repr(out["message"]))
    check("and marks it told",
          worlddoc.find_wanted(worlddoc.load(), "chicken packet")["announced"] is True)

    print("\n8. the camera is asked by name, and the budget grows with the list")
    seed()
    be = FakeBackend(QUIET_CAPTION)
    await LiveAgent(backend=be).tick("img")
    vp = next(p for k, p in be.prompts if k == "vision")
    check("both items in the vision prompt", "chicken packet" in vp and "onions" in vp)
    check("asked as one block, not two Q-indexed watches", vp.count("STANDING SEARCH") == 1)

    print("\n9. the backstop nudges when a request opened no search")
    worlddoc.clear()
    doc = worlddoc._empty_doc()
    doc["recent"] = [{"ts": time.time(), "text": "a caption"}]
    from server.live import triggers
    triggers.mark_user_turn(doc, "I need your help in finding the chicken and onion")
    agent = LiveAgent(backend=FakeBackend(QUIET_CAPTION))
    note = agent._unwatched_request_note(doc)
    check("nudge fires", "add_wanted" in note, repr(note))
    worlddoc.add_wanted(doc, ["onions"])
    check("silent once something is being watched",
          agent._unwatched_request_note(doc) == "")

    print("\n10. Stage 2 finally learns what the user said")
    doc = worlddoc._empty_doc()
    triggers.mark_user_turn(doc, "I need your help in finding the chicken and onion")
    p = LiveAgent(backend=FakeBackend(""))._build_speech_prompt(doc, "a caption", [], [])
    check("the words are in the prompt", "finding the chicken and onion" in p)
    check("labelled as a request", "[What they asked you" in p)

    print()
    if FAIL:
        print(f"FAILURES ({len(FAIL)}): " + "; ".join(FAIL))
        sys.exit(1)
    print("FIND LIST: ALL PASS")


asyncio.run(main())
