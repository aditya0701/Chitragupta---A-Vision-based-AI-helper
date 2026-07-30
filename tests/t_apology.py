import sys, os, asyncio
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
os.environ['TOOLS_ENABLED'] = 'true'
from server.agent.agent import ChitraguptAgent

def fresh():
    a = ChitraguptAgent.__new__(ChitraguptAgent)
    a._apology_streak = 0
    return a

fails = []
def ok(c, m):
    print(('PASS  ' if c else 'FAIL  ') + m)
    if not c: fails.append(m)

# ── 1. one apology is fine, two in a row is the loop ───────────────────────
a = fresh()
ok(a._is_stuck_apologising("I'm sorry, I was wrong about that.") is False,
   'first apology does NOT fire (courtesy is allowed)')
ok(a._is_stuck_apologising("I apologise again — I shouldn't have said that.") is True,
   'second consecutive apology DOES fire')
ok(a._is_stuck_apologising("My mistake once more.") is True,
   'third keeps firing until it actually answers')

# ── 2. a real answer between apologies resets the streak ───────────────────
a = fresh()
a._is_stuck_apologising("I'm sorry, I was wrong.")
ok(a._is_stuck_apologising("The dal is on the top shelf behind the loaf pan.") is False,
   'a real answer resets the streak')
ok(a._is_stuck_apologising("Sorry, I was mistaken again.") is False,
   'after a reset, one apology is allowed again')

# ── 3. THE IMPORTANT ONE: a useful correction must never be treated as a loop
a = fresh()
a._is_stuck_apologising("I'm sorry, I was wrong.")
useful = [
    "Sorry, you're right — those are black eyed beans, not toor dal. The toor dal is on the top shelf.",
    "My mistake. Take the pan off the heat now.",
    "I apologise — I misread it. It says 180C, so set the oven to that.",
    "Sorry about that. Add the cumin seeds to the hot oil.",
]
for u in useful:
    a2 = fresh()
    a2._is_stuck_apologising("I'm sorry, I was wrong.")   # streak = 1
    fired = a2._is_stuck_apologising(u)
    ok(fired is False, 'useful correction NOT treated as a loop: ' + u[:52] + '...')

# ── 4. non-apology replies never start a streak ────────────────────────────
a = fresh()
for t in ["The water is boiling — add the pasta.",
          "That package lists gelatin, so it isn't vegetarian.",
          "Nothing has changed yet."]:
    ok(a._is_stuck_apologising(t) is False, 'plain reply does not fire: ' + t[:40])
ok(a._apology_streak == 0, 'streak stays at 0 for plain replies')

# ── 5. a very long apologetic essay is out of scope (cap) ──────────────────
a = fresh()
a._is_stuck_apologising("I'm sorry, I was wrong.")
ok(a._is_stuck_apologising("I am so sorry. " + "Here is a great deal of further detail. " * 20) is False,
   'long reply is not counted as a bare apology')

# ── 6. corrective call resets the streak so it cannot re-fire ──────────────
class FakeBackend:
    async def chat(self, image_base64, prompt, think=False, **kw):
        class R: text = "The toor dal is on the top shelf, behind the loaf pan."; reasoning=""; model="m"; provider="p"; truncated=False; tool_calls=[]
        return R()

async def m():
    a = fresh()
    a.backend = FakeBackend()
    a._record_debug_step = lambda *x, **k: None
    a._is_stuck_apologising("I'm sorry.")
    a._is_stuck_apologising("I'm sorry again.")
    ok(a._apology_streak == 2, 'streak reached 2 before correction')
    out = await a._answer_not_apologise("where is the dal?", "a shelf with bags", [], "test")
    ok(out and 'top shelf' in out, 'corrective call returns an actual answer')
    ok(a._apology_streak == 0, 'corrective call resets the streak')
asyncio.run(m())

print('\n' + (f'{len(fails)} FAILURE(S)' if fails else 'APOLOGY LOOP: ALL PASS'))
sys.exit(1 if fails else 0)
