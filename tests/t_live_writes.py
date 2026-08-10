"""Do tool calls actually land when reasoning runs outside the lock?

The overlap harness (t_live_parallel.py) proves a user turn isn't blocked, but
it drives a backend that calls no tools — so it exercises the fast path where
`_apply` short-circuits before touching the lock at all. This one drives tool
calls through the deferred path end to end and reads the result back OFF DISK,
which is the thing that would break if a write window reloaded over the top of
its own mutation.

Also covers the propose-then-commit lifecycle and the mid-tick yield.

    python tests/t_live_writes.py
"""
import asyncio, pathlib, sys, tempfile
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')

from server.live import worlddoc
worlddoc.DOC_FILE = pathlib.Path(tempfile.mkdtemp()) / "w.json"
from server.backends import VisionResponse
from server.live.agent import LiveAgent

FAIL = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if not cond else ""))
    if not cond: FAIL.append(label)


class FB:
    SPLIT_VISION_REASONING = True
    SUPPORTS_NATIVE_TOOLS = True
    def __init__(self): self.script = []
    async def vision(self, image_base64, prompt="", max_tokens=160):
        return "User is holding a bag of dal."
    async def chat(self, image_base64=None, prompt="", conversation_history=None,
                   think=True, tools=None):
        calls = self.script.pop(0) if self.script else []
        return VisionResponse(text="ok", model="fake", provider="fake",
                              truncated=False, tool_calls=calls)


async def main():
    worlddoc.clear()
    a = LiveAgent(backend=FB())

    # 1. A chat turn that proposes a plan.
    a.backend.script = [[{"name": "propose_plan", "arguments": {
        "title": "Toor dal",
        "steps": [{"content": "Soak the dal"}, {"content": "Pressure cook"}]}}]]
    out = await a.chat("help me make toor dal")
    doc = worlddoc.load()
    check("deferred tool call persisted to disk", worlddoc.get_proposal(doc) is not None)
    check("proposal did not leak into tasks", doc["tasks"] == [], repr(doc["tasks"]))
    check("proposal shows in returned render", "[PROPOSED PLAN" in out["doc"])
    check("tool_calls reported back", [t["tool"] for t in out["tool_calls"]] == ["propose_plan"])
    check("raised_ts stamped by the chat turn", doc["proposal"]["raised_ts"] > 0)

    # 2. A tick that logs an environment fact — the deferred path on tick().
    a.backend.script = [[{"name": "log_environment", "arguments": {"fact": "dal is on the counter"}}],
                        []]   # stage 2 speech call
    out = await a.tick("img")
    doc = worlddoc.load()
    check("tick tool call persisted", any("dal is on the counter" in f["fact"]
                                          for f in doc["environment"]))
    check("tick caption recorded", doc["recent"][-1]["text"].startswith("User is holding"))
    check("tick reported its tool call", [t["tool"] for t in out["tool_calls"]] == ["log_environment"])

    # 3. The user says yes.
    a.backend.script = [[{"name": "commit_plan", "arguments": {}}]]
    await a.chat("yes go ahead")
    doc = worlddoc.load()
    check("commit promoted the plan", [t["content"] for t in doc["tasks"]]
          == ["Soak the dal", "Pressure cook"])
    check("proposal cleared after commit", worlddoc.get_proposal(doc) is None)

    # 4. A tick fired while the user is waiting must yield without reasoning.
    before = len(worlddoc.load()["recent"])
    a._user_waiting = True
    a.backend.script = [[{"name": "log_environment", "arguments": {"fact": "SHOULD NOT HAPPEN"}}]]
    out = await a.tick("img")
    a._user_waiting = False
    doc = worlddoc.load()
    check("yielded tick is flagged", out.get("yielded") is True)
    check("yielded tick still kept its caption", len(doc["recent"]) == before + 1,
          f"{before} -> {len(doc['recent'])}")
    check("yielded tick ran no tools",
          not any("SHOULD NOT" in f["fact"] for f in doc["environment"]))

    print("\n" + ("ALL DEFERRED-WRITE CHECKS PASSED" if not FAIL
                  else "FAILURES: " + ", ".join(FAIL)))

asyncio.run(main())
sys.exit(1 if FAIL else 0)
