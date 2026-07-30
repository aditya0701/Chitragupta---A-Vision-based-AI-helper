import sys, os
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
os.environ['TOOLS_ENABLED'] = 'true'
from server.agent.agent import ChitraguptAgent

a = ChitraguptAgent.__new__(ChitraguptAgent)
f = a._is_silent_live_reply

# --- MUST be suppressed (silence narration) ---
SILENT = [
    # the exact 2026-07-28 leak
    "Nothing has visibly changed in the scene. I'll stay quiet until there's "
    "something new to report.",
    "Nothing has visibly changed.",
    "Nothing changed.",
    "Nothing much has changed since the last frame.",
    "I'll stay quiet until something happens.",
    "Staying quiet for now.",
    "I'll remain silent.",
    "Silent as instructed.",
    "Nothing relevant to report.",
    "Nothing new.",
    "Nothing to report.",
    "Nothing worth mentioning right now.",
    "No changes.",
    "No relevant updates.",
    "The scene remains the same.",
    "Everything is still unchanged.",
    "[SILENT]",
    "",
    "   ",
]

# --- MUST NOT be suppressed (real content the user needs) ---
SPEAK = [
    # the guard cases: no-change wording + something substantive
    "Nothing has changed with the heat, but the onions are starting to brown — "
    "give them a stir.",
    "Nothing else has changed, but the timer shows 2 minutes left.",
    "No change to the pan, but add the garlic now.",
    "Nothing new on the label, but it does list 3 allergens.",
    # ordinary substantive updates
    "The water is boiling — add the pasta.",
    "That package lists gelatin, so it isn't vegetarian.",
    "The oil is shimmering. Add the cumin seeds now.",
    # long substantive reply that merely mentions the phrase
    "Nothing has changed with the dal itself, " + "and here is the rest of what "
    "I can see in some detail across the counter and the stove and the shelf "
    "behind it, which is a lot of information for one reply. " * 2,
]

fails = []
for t in SILENT:
    if not f(t):
        fails.append(("LEAKED (should be silent)", t))
for t in SPEAK:
    if f(t):
        fails.append(("SWALLOWED (should be spoken)", t))

for kind, t in fails:
    print(f"{kind}: {t[:80]!r}")

print(f"\nsilent cases: {len(SILENT)}, speak cases: {len(SPEAK)}, failures: {len(fails)}")
assert not fails, f"{len(fails)} failure(s)"
print("SILENCE DETECTION: ALL PASS")
