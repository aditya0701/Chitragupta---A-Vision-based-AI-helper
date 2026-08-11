# The find list — design

**Status:** implemented 2026-08-11. Verified by `tests/t_live_wanted.py`
(10 scenarios, end to end through the real `tick()` and `poll()`).
Not yet run against live traffic.
**Fixes:** the 2026-08-10 session, where the camera found the chicken at
`11:39:42`, the caption said so in plain English, and the user was told nothing
until they asked 65 seconds later.

---

## 1. The failure this exists to prevent

Three links have to hold between "the user asks for something" and "the user is
told it was found." In the session, two of them broke.

```
1. DeepSeek poses the question to the camera   →  SKIPPED
2. The camera answers it                       →  WORKED
3. The finding reaches the user as speech       →  REFUSED
```

**Link 1** — `set_expectation` was never called, not once. DeepSeek reached for
`set_vision_focus` and wrote the search into the standing brief as prose:
*"…is likely gathering the onions, tomatoes…"*. That is the soft nudge that lost
the black-eyed beans. It half-worked here only because the brief happened to
contain the word "onions".

**Link 2** — the camera reported *"boneless chicken breast is present and within
reach for the curry step; onions are not visible here"*. Correct, located, and
it volunteered a negative. Nothing was lost in the caption.

**Link 3** — that sentence went straight into Stage 2's
`[What the camera just reported]` block, and Stage 2 chose `[SILENT]`. It had no
idea anyone had asked about chicken: `_build_speech_prompt` passes
`[User last spoke] 143s ago` and **never the words**.

Every gate upstream was open. `in_followup_window` was true (143 s of a 180 s
window). `tool_results` was non-empty. The politeness budget would have been
bypassed. Stage 2 was asked the question and answered no, correctly, given what
it was shown.

**So the design principle:** finding is an observation, and observations must
not be judgement calls. Detection becomes arithmetic; only the *wording* stays
with a model.

---

## 2. The keystone

> **There is no tool for marking something found.**

The model cannot claim a find. `add_wanted` opens a search; nothing lets DeepSeek
write `status="found"`. Only the caption parser sets it, and only from an
explicit labelled answer produced by the thing that actually saw the pixels.

This is what structurally prevents the beans failure — *"several bags of
lentils"* upgraded into *"I can see the beans"*. An inference can no longer
reach the found state, because the found state has no model-facing door.

---

## 3. Schema

New top-level key in `worlddoc._empty_doc()`, alongside `expectations`:

```python
"wanted": [
    {
        "id": "w1",
        "item": "onions",              # short noun phrase, sent to the camera verbatim
        "asked_ts": 1754820000.0,
        "asked_text": "I need your help in finding the chicken and onion",
        "status": "open",              # open | found | dropped
        "found_ts": None,
        "where": None,                 # the FOUND payload: location + any label text
        "misses": 0,                   # NOT VISIBLE count
        "unclears": 0,                 # UNCLEAR count
        "announced": False,            # has the user actually been told
        "stuck_raised": False,         # has the give-up question been asked
    }
]
```

Plus, in the same commit and independent of everything else:

```python
"last_user_msg": "",                   # the words, not just last_user_turn_ts
```

### Why `announced` is separate from `status="found"`

One flag, one consequence. `found` is an observation about the world;
`announced` is a fact about speech. If the utterance fails — network error, the
model returns garbage, the tick dies — the item is still found and the user is
still owed it. Fusing them would make a failed utterance indistinguishable from
a delivered one, which is the same class of bug as v1's `found` flag driving
three unrelated outcomes.

The consequence is a **self-healing announcement**: the trigger condition is
`status == "found" and not announced`, which is pure doc state. It re-fires on
the next tick — or the next `poll()` — until the words actually land.

---

## 4. Lifecycle

```
  user asks
      │
      ▼
 [1] add_wanted            status=open
      │
      ▼
 [2] one question per tick  ──►  camera answers, labelled by item name
      │
      ▼
 [3] fold_wanted(doc, caption)   parser only. status=found, where=<payload>
      │                          NO MODEL. NO JUDGEMENT.
      ▼
 [4] triggers.check(doc)    emits wanted_found for found-and-unannounced
      │
      ▼
 [5] forced announcement    Stage 2 writes the words; SILENT is not on the menu
      │                     deterministic fallback if it misbehaves
      ▼
 [6] announced=True         item leaves the question list
      └──► auto-promote `where` into a durable environment fact
```

---

## 5. Link 1 — opening the list, non-electively

### `add_wanted(items: list[str], because: str)`

Takes a **list**, so "find the chicken and onions" is one call, not two.
Bounded by `MAX_WANTED` (6) — beyond that the camera cannot scan for everything
in one frame.

`SYSTEM_BRIEF` gains one rule:

> If the user asks you to find, locate, spot or keep an eye out for anything,
> you MUST call `add_wanted` before you reply. Telling them you will watch for
> it is not watching for it. There is no other way to start a search.

### The backstop that makes it self-healing

Prompt rules get ignored. So `last_user_msg` earns its second use: when the tick
prompt is built, if there is a stored user request and no `wanted` item covering
it, render a nudge:

```
[Outstanding request — 40s ago, nothing is being watched for]
"I need your help in finding the chicken and onion"
If this is a request to find something, call add_wanted now.
```

A missed opening is recovered on the *next tick* rather than lost for the
session. This is deliberately not a server-side regex over the user's words —
parsing intent from natural language is brittle, and a stale doc that re-asks is
cheaper than a wrong parse that opens a search for nothing.

### `drop_wanted(item)`

"Never mind, found them myself." Also called when the user answers a `stuck`
question with a location that resolves it conversationally.

---

## 6. Link 2 — one question, whatever the list length

The current design sends one question per watch and caps at
`MAX_ACTIVE_BRIEFS` (4). Three finds plus a safety watch fills it, and a fifth
watch silently never reaches the camera — which under a find-driven trigger
becomes a permanent blind spot on something explicitly asked for.

Instead, the whole list becomes **one synthesized question occupying one slot**,
prepended in `_vision_questions`:

```
Q1: Which of these are visible in this frame — onions, tomatoes, chicken packet?
```

with a format block that answers **by name, one line per item**:

```
Answer one line per item, naming the item exactly as written above:
  onions: FOUND — <exactly where in the frame, plus any label text you can read>
  onions: NOT VISIBLE — <what is in that part of the frame instead>
  onions: UNCLEAR — <what you can make out, and what blocks a confident answer>
```

Three properties fall out:

- **Cost stops scaling with item count** on the slot budget. One slot for one
  item or for six. Reply-token budget still scales:
  `VISION_MAX_TOKENS + VISION_TOKENS_PER_QUESTION * (n_questions + n_wanted)`.
- **No `Q<n>` index.** This kills the identity hazard: `Q<n>` is positional,
  built by `enumerate(questions, 1)`, over a list `_vision_questions` rebuilds
  every tick from `open_expectations` — filtered by in-progress tasks and
  priority-sorted. Close one watch and everything renumbers; add a high-priority
  safety watch mid-search and it sorts to the front and shifts every index. The
  vision call runs *unlocked* for ~1.5 s, so the doc can move underneath it.
  Parsing `Q2: FOUND` against a re-derived list will confidently announce
  "found the tomatoes" when the camera found the onions — wrong item, wrong
  location, stated as fact, then written into durable memory. Names have no such
  failure mode.
- **Two found in one frame is the normal case,** not an edge case.

The existing `NOT VISIBLE is a real, useful answer` and
`never guess an identification to be helpful` wording carries over verbatim.
The negative must stay an explicit result — a "no" that looks like silence is
how the beans were lost.

---

## 7. Link 3 — folding the answer in (no model, no tokens)

```python
def fold_wanted(doc: dict, caption: str) -> list[dict]:
    """Match labelled answer lines against open wanted items. Returns the items
    newly moved to found. Pure string matching — never a judgement."""
```

Called in **phase 3a**, inside the write window, immediately after
`add_recent(doc, caption)` at [agent.py:716] — `caption` is already in scope and
the doc is already locked and reloaded.

Rules:

- Match on the **exact item string we sent**, case-insensitive. Not positional,
  not fuzzy.
- A line naming something not on the list is ignored — the model invented it.
- `FOUND` → `status="found"`, `found_ts`, `where=<detail>`. This **claims** the
  item, same discipline as `expectation.status="fired"`, so a caption still
  sitting in `recent` cannot re-fire it.
- `NOT VISIBLE` → `misses += 1`.
- `UNCLEAR` → `unclears += 1`.
- An item absent from the answer entirely → **nothing.** Not a miss. Silence is
  not evidence, and counting it as one would let a truncated reply drive a
  give-up.

On `FOUND`, also promote the location into an environment fact
(`add_environment_fact`), because a found location *is* durable spatial memory
and it makes "where did I put the onions" answerable later for free.

---

## 8. Link 4 — the trigger

`triggers.check(doc)` **keeps its current signature.** The caption was already
consumed in phase 3a, so this stays pure doc arithmetic — which means it works
unchanged on the `poll()` path at [agent.py:983], where there is no frame at all.
A found-but-unannounced item gets announced on the next heartbeat even if the
camera has gone dark.

Two new kinds:

| Kind | Fires when | Priority |
|---|---|---|
| `wanted_found` | any item has `status=="found" and not announced` | normal |
| `wanted_stuck` | `misses >= WANTED_STUCK_ASKS` or `unclears >= WANTED_UNCLEAR_ASKS`, and not `stuck_raised` | low |

**`wanted_found` coalesces.** *One* event carrying *all* currently-unannounced
found items, never one event per item. Three items found in one pantry shot
produce one utterance — *"Onions are in the wire basket by the radiator,
tomatoes in the bowl just behind them"* — not three.

Priority stays `normal`. It does not need `high`: phase 3e's `important` check
already includes `bool(events)`, so any trigger bypasses the politeness budget.
`high` means physical risk, and inflating it here would be exactly the
one-flag-many-consequences mistake.

**`wanted_stuck`** is the answer to a search that never resolves — the item that
comes back `NOT VISIBLE` for four minutes, or `UNCLEAR` every time because of a
bad angle. It converts a silent dead end into something actionable:

> "I still haven't spotted the tomatoes — where do you usually keep them?"
> "I keep half-seeing something on the left shelf that might be the tomatoes;
> hold steady on it for a second?"

Different wording from different causes, but **one consequence** — ask the user
for help. `stuck_raised` claims it so it asks once, not every tick.

---

## 9. Link 5 — forced announcement

Phase 3d currently runs one prompt that offers `[SILENT]` as a valid answer.
When a `wanted_found` event is present, it runs a **different prompt** instead:

```python
def _build_announce_prompt(self, doc, caption, found_items) -> str:
    """The user asked for these and the camera has now seen them. This is not
    a decision about whether to speak — it is the wording of speech that is
    already going to happen. [SILENT] is not offered."""
```

It carries: what they asked (`asked_text`), each item, and each `where` payload.
It asks for one or two spoken sentences and nothing else.

### The fallback that makes "forced" actually forced

If the announce prompt returns `[SILENT]`, empty, or errors, the server composes
the sentence itself from the `where` payloads:

```
Found the onions — wire basket, left of the radiator.
```

Plain, deterministic, no model involved. **The user is told even when the model
misbehaves.** A forced path that can still be talked out of by the thing it was
built to overrule is not forced.

Phase 3e then marks `announced=True` on exactly the items in the event — and
only if `text` actually survived to be spoken. If it was suppressed, `announced`
stays `False` and the trigger re-fires next tick.

---

## 10. The independent smaller fix

Ship this first; it is three lines and it covers every case where no search was
ever opened.

`triggers.mark_user_turn(doc, text)` stores `last_user_msg`, and
`_build_speech_prompt` renders it directly above the SPEAK/STAY-SILENT rules:

```
[What they asked you, 143s ago]
"I need your help in finding the chicken and onion"
```

The existing SPEAK rule already reads *"they asked you for something and this
answers it"* — it has simply never had an operand. Carry any explicit promise
the assistant made the same way; an outstanding promise is a debt and Stage 2
should see it.

---

## 11. Config

```python
MAX_WANTED            = 6    # items the camera can scan for in one frame
WANTED_STUCK_ASKS     = 40   # NOT VISIBLE count before asking the user (~4 min)
WANTED_UNCLEAR_ASKS   = 8    # UNCLEAR count before asking them to steady the camera
```

## 12. Render

New section, placed between `[Open expectations]` and `[Earlier this session]`
— it changes more often than tasks and less often than `recent`, which keeps the
stability-first prefix-cache ordering intact:

```
[Looking for]
- onions — not spotted yet (asked 13:37, 22 frames checked)
- tomatoes — unclear 6 times; camera cannot get a confident look
- chicken packet — FOUND 13:39, red vacuum pack on the second fridge shelf ✓ told
```

## 13. Tools

| Tool | Purpose |
|---|---|
| `add_wanted(items, because)` | opens a search. The only entry point |
| `drop_wanted(item)` | cancel — "never mind", or resolved conversationally |

Nothing marks an item found. See §2.

---

## 14. Implementation order

Each step is independently shippable and independently useful.

1. **`last_user_msg` → Stage 2 prompt.** Three lines. Fixes the general case
   including searches that were never opened. Do this first regardless.
2. **Schema + render + `add_wanted` / `drop_wanted` + `SYSTEM_BRIEF` rule.**
   The list exists and is visible in the doc; nothing acts on it yet.
3. **The single list question in `_vision_questions` + the by-name answer format
   in `vision.py`.** Answers start coming back; still nothing acts on them.
4. **`fold_wanted` + the `[Outstanding request]` backstop.** Detection works.
   Findings land in the doc silently. Verify against an exported session before
   wiring speech.
5. **`wanted_found` trigger + forced announce + deterministic fallback.** The
   user gets told. This is the step that fixes the reported bug.
6. **`wanted_stuck`.** Dead-end searches become questions.

Steps 1 and 5 are the ones that fix the session. 2–4 are what make 5 safe.

---

## 15. Open questions

- **Does an announced item stay in `wanted` or move out?** Keeping it makes
  "where were the onions again?" answerable from structured state rather than
  from prose. Leaning keep, with `MAX_WANTED` counting only `status=="open"`.
- **Should `wanted` survive a goal change?** A find opened during curry prep is
  probably stale once the goal changes. Leaning drop-on-`commit_plan`-of-a-new-goal.
- **Interaction with `set_expectation(anchor="event")`.** Two mechanisms now
  put questions to the camera. Event-anchored expectations are for *conditions*
  ("has the oil started to shimmer"); `wanted` is for *objects*. If that line
  blurs in practice, `wanted` should probably absorb the object-shaped ones —
  but not before there is evidence, since expectations also carry `task_id` and
  priority that `wanted` deliberately does not.
