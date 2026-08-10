"""Does a user turn actually overtake an in-flight tick?

Fake backend with controllable delays — no network, no real models. Measures
the wall-clock gap between a chat being submitted and its reasoning starting,
while a tick is mid-flight.

Two scenarios, because they used to fail for different reasons:

  A. the user speaks while the tick is being CAPTIONED. Fixed earlier, by
     moving the vision call outside the lock.
  B. the user speaks while the tick is REASONING. This one serialized fully
     until the lock was narrowed to cover writes only — the tick held the
     document across both DeepSeek round trips and any web_search they made,
     so a question arriving at that moment queued behind all of it.

B is the case that matters in a real session: at a 4s interval with a 1.5s
caption, most of the wall clock is spent in reasoning, so most questions
arrive during it.

    python tests/t_live_parallel.py
"""
import asyncio, pathlib, sys, tempfile, time
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')

from server.live import worlddoc
worlddoc.DOC_FILE = pathlib.Path(tempfile.mkdtemp()) / "w.json"
from server.backends import VisionResponse
from server.live.agent import LiveAgent

VISION_S, REASON_S = 1.5, 2.5
FAIL = []


class FakeBackend:
    SPLIT_VISION_REASONING = True
    SUPPORTS_NATIVE_TOOLS = True

    def __init__(self, mark):
        self.mark = mark

    async def vision(self, image_base64, prompt="", max_tokens=160):
        self.mark("vision start")
        await asyncio.sleep(VISION_S)
        self.mark("vision end")
        return "a caption"

    async def chat(self, image_base64=None, prompt="", conversation_history=None,
                   think=True, tools=None):
        who = "CHAT reasoning" if "[User says]" in prompt else "tick reasoning"
        self.mark(f"{who} start")
        await asyncio.sleep(REASON_S)
        self.mark(f"{who} end")
        return VisionResponse(text="ok", model="fake", provider="fake",
                              truncated=False, tool_calls=[])


async def scenario(label, speak_at, expect_under):
    """Run one tick, interrupt it at `speak_at`, report how long the user waited."""
    events = []
    t0 = time.perf_counter()
    mark = lambda what: events.append((round(time.perf_counter() - t0, 2), what))

    worlddoc.clear()
    agent = LiveAgent(backend=FakeBackend(mark))

    tick = asyncio.create_task(agent.tick("img"))
    await asyncio.sleep(speak_at)
    mark("USER SPEAKS")
    chat = asyncio.create_task(agent.chat("where are the onions?"))
    await asyncio.gather(tick, chat)

    print(f"\n  {label}   (user speaks at {speak_at}s)")
    for t, what in events:
        print(f"    {t:5.2f}s  {what}" + ("   <<<" if "USER" in what else ""))

    spoke = next(t for t, w in events if w == "USER SPEAKS")
    served = next(t for t, w in events if w == "CHAT reasoning start")
    waited = served - spoke
    serial = VISION_S + REASON_S - speak_at   # what full serialization would cost

    ok = waited < expect_under
    print(f"    user waited {waited:.2f}s   (serialized would be {serial:.2f}s, "
          f"budget {expect_under:.2f}s)")
    print(f"    {'PASS' if ok else 'FAIL'} — {100 * (1 - waited / serial):.0f}% less waiting")
    if not ok:
        FAIL.append(label)


async def main():
    print(f"  vision={VISION_S}s  reasoning={REASON_S}s")

    # A: mid-caption. The lock is free the whole time the vision call runs.
    await scenario("A. interrupt during VISION  ", speak_at=0.4, expect_under=0.5)

    # B: mid-reasoning. The tick is past phase 3a and inside its unlocked
    # Stage 1 call; the only lock it takes from here is a write window, so the
    # user should be served in milliseconds, not after the reasoning finishes.
    await scenario("B. interrupt during REASONING", speak_at=2.0, expect_under=0.5)

    print("\n" + "=" * 60)
    print("ALL OVERLAP CHECKS PASSED" if not FAIL else "FAILURES: " + ", ".join(FAIL))

asyncio.run(main())
sys.exit(1 if FAIL else 0)
