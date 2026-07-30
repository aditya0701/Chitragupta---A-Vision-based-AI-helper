import sys, os, asyncio, tempfile
from pathlib import Path
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
os.environ['TOOLS_ENABLED'] = 'true'

from server.agent import tasklist as T, build_default_tools
T.DOCUMENT_FILE = Path(tempfile.mkdtemp()) / "document.json"

from server.agent.agent import ChitraguptAgent

reg = build_default_tools()
assert reg.get("retract_observation"), "tool not registered"
print("registered:", [t.name for t in reg.list_tools()])

# Schema is valid for native function calling
schema = reg.get("retract_observation").to_openai_tool()
assert schema["function"]["parameters"]["required"] == ["item", "note_match"], schema
print("schema required:", schema["function"]["parameters"]["required"])

a = ChitraguptAgent.__new__(ChitraguptAgent)
a.tools = reg


class FakeBackend:
    SPLIT_VISION_REASONING = True
    SUPPORTS_NATIVE_TOOLS = True
    calls = 0

    async def vision(self, image_base64, prompt, max_tokens):
        FakeBackend.calls += 1
        return f"caption {FakeBackend.calls}"


a.backend = FakeBackend()


async def main():
    # 1. retract_observation runs through the native executor
    T.start_find_task("toor dal")
    T.add_observation("Find toor dal", "toor dal in the plastic bag", found=True)
    r = await a._run_structured_tool_calls([
        {"id": "1", "name": "retract_observation",
         "arguments": {"item": "Find toor dal", "note_match": "toor dal", "reopen": True}},
    ])
    print("\nnative  ->", r[0]["result"][:80])
    assert T.get_document()["items"][0]["status"] == "in_progress"

    # 2. and through the regex executor
    T.add_observation("Find toor dal", "wrong fact again")
    txt = 'x\n```tool\n{"name":"retract_observation","arguments":{"item":"Find toor dal","note_match":"wrong fact"}}\n```'
    r = await a._execute_tool_calls(txt)
    print("regex   ->", r[0]["result"][:80])
    assert T.get_document()["items"][0]["observations"] == []

    # 3. the vision stage actually charges the fine budget
    T.set_document("Shop", [{"content": "Read the label", "status": "in_progress",
                             "detail": "fine", "watch_for": "read the ingredients"}])
    assert T.active_detail() == "fine"
    for i in range(T.MAX_FINE_FRAMES_PER_ITEM):
        await a._run_vision_stage("FAKEB64", [], prev_caption="earlier caption")
    used = T.get_document()["items"][0]["fine_frames_used"]
    print(f"\nvision stage charged {used} fine frames; active_detail now "
          f"{T.active_detail()!r}")
    assert used == T.MAX_FINE_FRAMES_PER_ITEM, used
    assert T.active_detail() == "coarse", "vision stage did not bound the cost"

    # 4. a coarse frame must not charge anything
    before = T.get_document()["items"][0]["fine_frames_used"]
    await a._run_vision_stage("FAKEB64", [], prev_caption=None)
    assert T.get_document()["items"][0]["fine_frames_used"] == before
    print("coarse frames do not charge: OK")

asyncio.run(main())

# 5. prompt guidance mentions the new tool + the corrected-caption clause
a2 = ChitraguptAgent.__new__(ChitraguptAgent)
a2.tools = reg
a2.backend = FakeBackend()
p = a2._build_reason_prompt("what now?", scene="a pan", has_image=True)
assert "retract_observation" in p, "no guidance for the new tool"
print("\nprompt guidance present: OK")

print("\nINTEGRATION: ALL PASS")
