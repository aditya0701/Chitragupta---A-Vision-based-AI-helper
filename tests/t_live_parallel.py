"""Does a user turn actually overtake an in-flight tick?

Fake backend with controllable delays — no network, no real models. Measures
the wall-clock gap between a chat being submitted and its reasoning starting,
while a tick is mid-flight.
"""
import asyncio, pathlib, sys, tempfile, time
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')

from server.live import worlddoc
worlddoc.DOC_FILE = pathlib.Path(tempfile.mkdtemp()) / "w.json"
from server.backends import VisionResponse
from server.live.agent import LiveAgent

VISION_S, REASON_S = 1.5, 2.5
events = []
T0 = time.perf_counter()
def mark(what):
    events.append((round(time.perf_counter() - T0, 2), what))


class FakeBackend:
    SPLIT_VISION_REASONING = True
    SUPPORTS_NATIVE_TOOLS = True

    async def vision(self, image_base64, prompt="", max_tokens=160):
        mark("vision start")
        await asyncio.sleep(VISION_S)
        mark("vision end")
        return "a caption"

    async def chat(self, image_base64=None, prompt="", conversation_history=None,
                   think=True, tools=None):
        who = "CHAT reasoning" if "[User says]" in prompt else "tick reasoning"
        mark(f"{who} start")
        await asyncio.sleep(REASON_S)
        mark(f"{who} end")
        return VisionResponse(text="ok", model="fake", provider="fake",
                              truncated=False, tool_calls=[])


async def main():
    agent = LiveAgent(backend=FakeBackend())

    tick = asyncio.create_task(agent.tick("img"))
    await asyncio.sleep(0.4)                      # user speaks mid-tick
    mark("USER SPEAKS")
    chat = asyncio.create_task(agent.chat("where are the onions?"))
    await asyncio.gather(tick, chat)

    print(f"  vision={VISION_S}s  reasoning={REASON_S}s\n")
    for t, what in events:
        bar = "" if "USER" not in what else "   <<<"
        print(f"  {t:5.2f}s  {what}{bar}")

    spoke = next(t for t, w in events if w == "USER SPEAKS")
    served = next(t for t, w in events if w == "CHAT reasoning start")
    waited = served - spoke
    serial = VISION_S + REASON_S - 0.4
    print(f"\n  user waited {waited:.2f}s before being served")
    print(f"  fully serialized would have been {serial:.2f}s")
    print(f"  {'PASS' if waited < serial / 2 else 'FAIL'} — "
          f"{100 * (1 - waited / serial):.0f}% less waiting")

asyncio.run(main())
