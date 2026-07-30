import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
from server.agent import tasklist as T

T.DOCUMENT_FILE = Path(tempfile.mkdtemp()) / "document.json"

N = T.MAX_FINE_FRAMES_PER_ITEM
print("budget =", N)

T.set_document("Shopping", [
    {"content": "Check the broth label", "status": "in_progress", "detail": "fine",
     "watch_for": "Read the ingredient list"},
    {"content": "Get noodles", "status": "pending"},
])
assert T.active_detail() == "fine", "should start fine"

# Spend the budget one frame at a time
for i in range(1, N + 1):
    assert T.active_detail() == "fine", f"reverted early at frame {i}"
    T.record_fine_frame()
print(f"after {N} frames -> active_detail = {T.active_detail()!r}")
assert T.active_detail() == "coarse", "budget did not bound anything"

item = T.get_document()["items"][0]
assert item["detail"] == "coarse", "exhausted item still claims fine in the prompt"
assert item["fine_frames_used"] == N

# Further frames must not keep charging
T.record_fine_frame()
assert T.get_document()["items"][0]["fine_frames_used"] == N, "over-charged past budget"

# Survives a restart (re-read from disk, no in-memory state)
import importlib
doc = json.loads(T.DOCUMENT_FILE.read_text())
assert doc["items"][0]["fine_frames_used"] == N, "counter not persisted"
print("persisted across restart: OK")

# An unrelated full-replace must NOT refill the budget (the original failure
# was drift, and update_task_list runs constantly)
T.set_document("Shopping", [
    {"content": "Check the broth label", "status": "in_progress", "detail": "fine"},
    {"content": "Get noodles", "status": "in_progress"},
])
assert T.get_document()["items"][0]["fine_frames_used"] == N, "full-replace refilled budget"
assert T.active_detail() == "coarse", "full-replace re-armed fine"
print("full-replace does not refill: OK")

# An EXPLICIT new close-look request (coarse -> fine) re-arms it
T.set_document("Shopping", [
    {"content": "Check the broth label", "status": "in_progress", "detail": "coarse"},
])
T.set_document("Shopping", [
    {"content": "Check the broth label", "status": "in_progress", "detail": "fine"},
])
assert T.get_document()["items"][0]["fine_frames_used"] == 0
assert T.active_detail() == "fine"
print("explicit re-request re-arms: OK")

# Completing the step still ends the close look (scope bound intact)
T.set_document("Shopping", [
    {"content": "Check the broth label", "status": "completed", "detail": "fine"},
])
assert T.active_detail() == "coarse", "scope bound broken"
print("scope bound still holds: OK")

print("\nDETAIL BUDGET: ALL PASS")
