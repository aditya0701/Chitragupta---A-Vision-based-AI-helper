import sys, tempfile
from pathlib import Path
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
from server.agent import tasklist as T

T.DOCUMENT_FILE = Path(tempfile.mkdtemp()) / "document.json"

# --- Replay the 2026-07-28 failure exactly ---
T.start_find_task("toor dal")
T.add_observation("Find toor dal", "toor dal spotted in a plastic bag on the counter", found=True)
T.add_observation("Find toor dal", "toor dal bag is on an upper shelf, open and upright")

doc = T.get_document()
item = doc["items"][0]
print("before:", item["status"], item["observations"])
assert item["status"] == "completed", "find-goal should have completed"
assert len(item["observations"]) == 2

# The user says: "the thing in the plastic bag is black eyed beans not toor dal"
r = T.retract_observation("Find toor dal", "toor dal", reopen=True)
print("\nretract ->", r)

item = T.get_document()["items"][0]
print("after :", item["status"], item["observations"])
assert item["observations"] == [], "wrong observations survived the correction"
assert item["status"] == "in_progress", "wrongly-completed item stayed completed"

# The rendered prompt injection must no longer carry the false fact
rendered = T.render_summary(T.get_document())
print("\nrendered [Task list]:\n" + rendered)
assert "upper shelf" not in rendered, "false fact still riding along in the prompt"

# --- Partial match: only the wrong note goes ---
T.add_observation("Find toor dal", "black eyed beans confirmed in the plastic bag")
T.add_observation("Find toor dal", "toor dal still not located")
r = T.retract_observation("Find toor dal", "toor dal still not")
print("\npartial ->", r)
obs = T.get_document()["items"][0]["observations"]
assert obs == ["black eyed beans confirmed in the plastic bag"], obs
print("selective removal OK:", obs)

# --- Failure modes ---
print("\nno match  ->", T.retract_observation("Find toor dal", "gelatin")[:95])
print("bad item  ->", T.retract_observation("Nonexistent", "x")[:95])
print("empty arg ->", T.retract_observation("Find toor dal", "   ")[:95])
assert "No task list item matching" in T.retract_observation("Nonexistent", "x")

# --- note field cleared on reopen ---
T.set_document("Cook", [{"content": "Check broth", "status": "completed",
                         "note": "verified beef-free from the label"}])
T.retract_observation("Check broth", "beef-free", reopen=True)
it = T.get_document()["items"][0]
assert it["note"] is None and it["status"] == "in_progress", it
print("\nnote cleared + reopened OK")

print("\nRETRACTION: ALL PASS")
