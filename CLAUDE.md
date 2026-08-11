# Vision Chitragupta — v2

A hands-free, camera-equipped voice assistant for hands-on tasks — cooking,
repairs, shopping, anything where you're working and can't look at a screen.
Named for the Hindu record keeper who observes, records, and reports.

It watches through a phone camera, keeps a **world document** of everything it
has seen and decided, and speaks only when speaking is worth it. The user is
**listening, not reading** — that constraint drives most design choices.

> **`HANDOFF.md` is where to start** — current status and what to do next.
>
> **`DECISIONS.md` holds the failure log and the reasoning behind every design
> choice here.** Read the relevant section before changing an area — most of the
> non-obvious code is scar tissue from a specific bug.
>
> **v1 is superseded.** It still runs (`/`, `/v1/*`) and is not being developed.
> Its reference and failure log are archived at `docs/v1/`. Section references
> in code comments — `DECISIONS.md §6.3`, `DECISIONS.md 4.4` — point at
> **`docs/v1/DECISIONS.md`**, which is where those numbers live.

---

## Run it

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000     # from repo root
# then open  http://localhost:8000/          ← v2, the default
```

**Deployed:** <https://chitragupta-k6ek.onrender.com/>

**`/` serves v2.** `/live` still serves the same page and is kept as an alias
so older links and docs do not break. **v1 is hidden, not removed** — its UI is
at `/v1` and its API is unchanged on `/v1/*`; nothing links to it except one
header link out of v2.

- Requires `TOOLS_ENABLED=true`. `LIVE_BACKEND_MODE` defaults to `deepinfra`,
  so it only needs setting to override.
- **v2 refuses to start with vision on Groq** and says so at boot. If `/v2/*`
  returns "v2 refuses to start with vision on Groq", the fix is almost always
  *restart the server* — the config on disk was already right last time. §5.3.
- The startup line to check:
  `Initialized live agent | backend mode: … | VISION ON: … | reasoning: …`
- **Avoid `--reload`** — WatchFiles has served stale bytecode on Windows.
  Restart manually.
- Camera/mic need HTTPS or the literal hostname `localhost`. A bare LAN IP over
  HTTP fails the browser's secure-context check.
- The service worker **does not touch v2** — `/`, `/live`, `/static/live*` and
  `/v2/*` are excluded outright, so no `CACHE_NAME` bump is needed for ordinary
  v2 work. It was bumped exactly once for v2, to `v20`, when `/` changed hands:
  `/` was in `SHELL_URLS` and cached cache-first, so every browser that had ever
  loaded v1 would have kept being served v1's `index.html` at the new default
  address. v1's shell is now cached at `/v1` instead. `docs/v1/DECISIONS.md` 5.1.

---

## The shape of it

**The world document is primary state; speech is a side-effect.** Ticks update
the document continuously. A separate, zero-cost arithmetic engine decides when
the user is owed something. This inversion is the whole design — v1 asked "what
do I say about this frame?" on every frame, which is why it narrated.

```
Phone/browser ──► /v2/tick   (camera frame, on an interval)
                  /v2/chat   (the user said something)
                  /v2/poll   (heartbeat — arithmetic only, usually free)
                      │
                      ├─► [Vision] Qwen3-VL-30B on DeepInfra
                      │            frame + brief ──► plain-text caption
                      │            (never reasons, never decides)
                      │
                      ├─► [Reasoning] DeepSeek v4-flash — text only, never
                      │            sees pixels. Owns every decision and tool call.
                      │
                      ├─► [Triggers] pure arithmetic over the doc. No tokens.
                      │
                      └─► world document (+ optional speech)
```

### The tick, in five phases

Only the odd-numbered phases hold the lock.

| | | lock | ~cost |
|---|---|---|---|
| 1+2 | caption the frame | **unlocked** | ~1.5s, one vision call |
| 3a | fold the caption in, claim triggers | locked | ~1ms |
| 3b | **Stage 1** — bookkeeping reasoning, tools only | **unlocked** | ~2s |
| 3c | is the speech question even worth asking? | locked | ~1ms |
| 3d | **Stage 2** — the speech decision | **unlocked** | ~1s |
| 3e | politeness gate, record the utterance | locked | ~1ms |

Stage 1 is scored on one thing: *is the document now accurate?* Its prose is
discarded. Stage 2 gets a small, separate prompt — no tools, no system brief,
no full document — and answers one question: *does the user need to hear
something?* Splitting them fixed a failure reported three times, where one call
scored on both objectives always chose silence. §6.1.

An idle tick costs **one vision call and one reasoning call**, and phase 3d is
skipped entirely when nothing happened.

### The lock covers writes, not thinking

`LiveAgent._lock` is held only inside `_write_window()`: reload → mutate →
save → release, measured in milliseconds. **Every model call happens outside
one**, including the reasoning that produces the tool calls —
`_reason(deferred_writes=True)` hands them to `_apply`, which takes its own
window.

The cost of releasing it is that a turn reasons about a document that may have
moved by the time it writes. That's why every window **reloads** — never reuse
a doc read before a model call — and why the doc-mutating tools degrade to a
harmless "no match" string rather than raising. Losing a tick's bookkeeping to
a race is recoverable; the next frame re-derives it. Making the user wait is
not. §2.

---

## The world document

`server/data/live/worlddoc.json` — gitignored, survives restarts by design.
Rendered into every prompt by `worlddoc.render()`, in this order:

```
[Current time: HH:MM:SS]
[Goal] <title>
[Tasks]                              [ ] pending  [~] in_progress  [x] completed  [-] skipped
[PROPOSED PLAN — NOT COMMITTED]      ← only when awaiting the user
[Camera focus — fine frames, N used]
[Open expectations]                  time-anchored show a countdown; event-anchored show the watch
[Looking for]                        the find list — still open, or found and where
[Earlier this session]               compacted narrative, with time spans
[Known environment facts]            durable spatial memory
[Recent observations]                raw captions, newest last
```

**Section order is stability-first** — title/tasks/narrative/environment change
rarely, `recent` changes every tick — so DeepSeek's prefix cache gets the
longest possible unchanged prefix across consecutive ticks.

Every entry is timestamped in **the user's** zone (`DEFAULT_TIMEZONE`), not the
server's. Render runs UTC; a naive `fromtimestamp()` once stamped the whole
document — including the `[Current time]` header the model does all its
temporal arithmetic against — two hours off, so "will this be done by four?"
was answered about a different four.

`rev` is monotonic, bumped on write-window **entry** (entry order *is* write
order, since windows are serialized by the lock). Every response echoes it as
`doc_rev`; the client drops any render older than one it has already painted.

### Overflow

`recent` is bounded at `RECENT_MAX` (24). On overflow the oldest
`COMPACT_BATCH` (16) captions are summarised into `narrative` by one cheap
model call — **time spans preserved**, not just facts — plus any durable
environment facts worth promoting. Raw captions are never silently dropped, and
the freshest window is never compacted.

---

## Plans are proposed, not written

The split is **what it saw** vs **what it decided**.

Observations — captions, `log_environment`, `resolve_expectation`, focus
changes — write **silently and immediately**. Gating those would turn every
tick into a permission prompt.

A **plan** is a decision about how the user spends the next hour, made from a
photograph and a web search. Once it lands in `tasks` it is re-injected into
every later prompt as settled fact, and the model reads its own guess back as
memory. So:

```
propose_plan  → doc["proposal"], NOT tasks. Nothing tracks it, no expectations.
                Must be said out loud in the same reply — the user cannot see it.
commit_plan   → promotes it to real tasks. Called on assent, including implicit
                ("yes", "go on", or visibly starting step one).
discard_plan  → drops it.
```

An unanswered proposal re-raises every `PROPOSAL_RERAISE_S` (150s) — the
assistant is blocked on the user and the user has no idea. One proposal at a
time, replaced rather than appended. §3.

**A tick may only commit on visibly starting step one.** A frame cannot tell
you someone said yes.

---

## The vision stage

The reasoning model cannot see. It aims the camera two ways:

**`set_vision_focus(brief, detail, mode)`** — the standing lens. One or two
plain sentences saying *what the user is physically doing*. Replaced, never
appended, so duplicates are impossible by construction.

- The model writes **only the activity**. The grip/posture/danger instructions
  are attached automatically to every frame. When the model wrote the whole
  block it wrote checklists ("whether a drain pan is underneath"), which made
  every different-but-fine setup read back as a list of absent items. §4.2
- `mode="form"` → report posture, grip, danger, and anything else out of place.
  `mode="read"` → transcribe text verbatim, say what's illegible and how to fix
  it. Read mode forces `fine` and loads none of the form wording. Asked to read
  a packet in form mode, the camera reported "hand position is not visible"
  while the cooking instructions went untranscribed. §4.3
- `detail="fine"` (1024px) vs `"coarse"` (640px). Bounded by
  `MAX_FINE_FOCUS_FRAMES` (120) — a nudge-then-backstop, since v1 proved a
  fine mode set once is never voluntarily reverted.

**`set_expectation(anchor="event", condition=…)`** — a discrete watch. The
condition is put to the camera **verbatim as a question on every frame** until
resolved. Answers come back as `Q1: FOUND / NOT VISIBLE / UNCLEAR` lines, asked
*first*, before the description.

Both must ask for **observations, never judgement** — the reasoning stage
decides. A model asked "is the grip safe?" returns a reassuring guess; asked
"are the fingertips curled back or extended flat?" it returns a fact. §4.1

At most `MAX_ACTIVE_BRIEFS` (4) watches reach any one frame, high-priority
first, because the vision model cannot answer nine questions and describe the
scene inside one reply.

**Each tick is given the previous caption as text**, so it describes *change*
rather than re-describing the scene. It is also forbidden from describing the
camera itself — the user is holding it and knows they moved it. §4.4

---

---

## The find list

"Find the chicken and the onions" → `add_wanted(["chicken packet", "onions"])`.
Every open item is put to the camera **by name, in one block, on every frame**,
answered `<item>: FOUND / NOT VISIBLE / UNCLEAR`. One block, not one question
per item, so the list never competes with `MAX_ACTIVE_BRIEFS` and a fourth
search can't silently stop reaching the camera.

**Ask for an appearance, never a category.** `add_wanted` takes a `looks_like`
per item — *"small cream-white beans, each with a distinct black spot, in a
clear bag"*, not *"black-eyed beans"*. A camera cannot see what a thing **is**,
only what it looks like, so a category name gets you a guess: asked for
black-eyed beans it reported "several bags of lentils" and a find was claimed
off that. The vision prompt states the rule at the point of decision — *a bag
of something that could be the right kind of thing is UNCLEAR, not FOUND*. This
is the actual fix for that failure; everything below is containment. §8.1

**Two paths can mark a find, and neither is free of the other.**
`worlddoc.fold_wanted()` reads the camera's labelled `<item>: FOUND` answer —
zero tokens, no judgement. `mark_found(item, evidence, where)` lets the
reasoning model declare one from prose, which is what covers a caption that
answered in words, a drifted format, and judgements no label can express. Its
`evidence` **must appear verbatim in the current caption and is checked**; a
paraphrase is rejected, and the quote is stored and rendered so an exported
session shows what justified every find. §8.1

**`unmark_found(item, correction)` closes the loop.** A retraction that only
clears the flag cannot hold — the caption that caused the mistake is still in
`recent` and re-justifies it next tick. So the wrong *evidence*, the wrong
*location* and the user's words all go into `ruled_out`, which rides in the
vision prompt as `NOT this — already checked and rejected: …` and blocks both
find paths. Matching is substring **plus distinctive-word overlap**, because
the same wrong bag gets re-described in slightly different words. Tuned to err
toward "keep looking": a false match leaves the item open and the misses
climbing, which eventually asks the user. §8.4

**A find forces speech.** `wanted_found` routes to its own prompt
(`_build_announce_prompt`) that never offers `[SILENT]`, and if the model
declines or errors anyway the server speaks a deterministic sentence built from
the location the camera wrote. A forced path the model can talk its way out of
is not forced. §8.2

`found` and `announced` are **separate flags** — one is about the world, the
other about speech. The announce trigger tests doc state, not this tick's
caption, so a failed utterance simply re-fires next tick, and it fires from
`poll()` too: a find that lands just as the camera goes dark still arrives.

Matching is **by name, never by Q-index** — the index is positional over a list
rebuilt each tick while the vision call runs unlocked, so `Q2: FOUND` would
eventually announce one item's location under another item's name. §8.3

---

## Triggers and speech

`triggers.check(doc)` is pure arithmetic — no tokens — and returns:

| kind | fires when |
|---|---|
| `expectation_due` | a time-anchored expectation passed its deadline unresolved |
| `stale_task` | an `in_progress` task unmentioned for `STALENESS_S` (480s) |
| `proposal_pending` | a plan proposed and unanswered for `PROPOSAL_RERAISE_S` (150s) |
| `wanted_found` | a find-list item was seen and the user hasn't been told (coalesced — one event for all of them) |
| `wanted_stuck` | a search hit `WANTED_STUCK_ASKS` (40) misses or `WANTED_UNCLEAR_ASKS` (8) unclears — ask the user instead of failing silently |

Each **claims** its state on firing (same double-fire lesson as v1's timers).

Unprompted speech is gated by a politeness budget, `MIN_UNPROMPTED_GAP_S` (90s).
Two things bypass it:

- **`[URGENT]`** — physical risk, or work about to be ruined. Reaches the user
  immediately. One flag, one consequence: it bypasses the gate and nothing else.
- **The follow-up window**, `FOLLOWUP_WINDOW_S` (180s). Answering the user used
  to *reset* the politeness gap, which silenced exactly the follow-up they were
  waiting for — asked to find the onions, the assistant said "I'll point them
  out when I see them", and that reply gagged it for the whole 90s search. It
  found them at +27s, logged them silently, and said nothing until asked again.
  A recent request is the one moment a follow-up is *solicited*. §6.2

---

## Tools (13)

Doc-mutating tools close over the agent's current in-memory doc, so a turn's
tool calls and the agent's own writes can never interleave on disk.

| | |
|---|---|
| `propose_plan` · `commit_plan` · `discard_plan` | the approval cycle |
| `add_wanted` · `drop_wanted` | the find list — open and cancel a search |
| `mark_found` · `unmark_found` | declare a find from prose (evidence-checked), and undo a wrong one |
| `update_tasks` · `mark_task` | a plan the user is **already** working through |
| `set_expectation` · `resolve_expectation` | deadlines and camera watches |
| `set_vision_focus` | the standing lens |
| `log_environment` · `retract_environment_fact` | durable spatial memory, and undoing it |
| `web_search` · `fetch_page` · `calculate` | inherited from v1 unchanged |

**Deliberately absent:** `start_timer` (subsumed by a time-anchored
expectation, which also has a resolution path timers never had) and
`request_camera` / `request_live_search` (the live UI owns the camera; chat
turns attach the current frame client-side).

`retract_environment_fact` takes a `correction`, not just a deletion. The raw
captions that produced the wrong inference are still in `recent` and will
suggest it again on the very next tick — a hole in the fact list doesn't block
that; a durable "the bag on the pantry shelf is NOT toor dal" does.

---

## The client (`live.js`)

| | |
|---|---|
| `FRAME_DIM` | `{ coarse: 640, fine: 1024 }` — caps the **longest** side |
| `JPEG_QUALITY` | 0.85. Cost scales with resolution, not quality — quality is not a lever |
| `CAPTION_REUSE_MS` | 15000 — a chat turn reuses the last caption if the scene hasn't moved |
| `FLAT_FRAME_STDDEV` | 2.0 — below this the frame is blank, not merely unchanged |
| `POLL_INTERVAL_MS` | 20000 |

**The diff gate** (32×32 grayscale, mean absolute delta) is the main cost
control: if the scene hasn't meaningfully changed, no request leaves the
browser. The server echoes `frame_detail` on every response and the client
sizes its **next** capture from it — resolution discarded in the browser can't
be recovered, so the decision has to run one frame ahead.

**Blank-frame detection runs before the diff gate.** A black frame *is* an
unchanged frame to a delta comparison, so a dead camera reported "nothing
changed" indefinitely. Standard deviation is the liveness test. §7.2

**`tickBusy` and `chatBusy` are separate flags** with separate queues.

---

## Hard rules

Each of these cost a real debugging session.

| Rule | Why |
|---|---|
| **Never `await` a model call inside a write window** | It holds `LiveAgent._lock` and re-serializes ticks and chat, undoing the whole phase split. `compaction.compact` is the one deliberate exception. §2.1 |
| **Every write window reloads** | Reusing a doc read before a model call silently rolls back whoever wrote in the meantime. §2.2 |
| **Never gate a chat send on `tickBusy`** | One shared `busy` flag in `live.js` kept the browser from sending a question until the tick returned — defeating every server-side overlap. §7.3 |
| **Apply a doc render only if `doc_rev` is newer** | Concurrent turns reply out of order; a slow tick's render predates a chat's writes and will stamp over it. §7.4 |
| **A brief must ask for observations, not judgement** | "Is the grip safe?" gets a reassuring guess. §4.1 |
| **Brief the camera on appearance, not category** | It cannot see what a thing *is*. "Black-eyed beans" got "several bags of lentils" read back as a find. §8.1 |
| **A model-declared find must quote the caption** | `mark_found` validates `evidence` against the real text and stores it, so every find is auditable in the export. §8.1 |
| **A retraction must record the wrong evidence, not just the correction** | They share no words, so the next tick re-found it from the same caption. §8.4 |
| **Stage 2 gets the user's words, not just a timestamp** | It was asked "does this answer what they wanted?" while never being shown what they wanted. Cost 65s with the chicken on screen. §8.2 |
| **Never add a pre-filter whose "no" looks like silence** | Cost a whole class of silently dropped frames. `docs/v1/DECISIONS.md` §6.2 |
| **One flag, one consequence** | `found` drove three unrelated outcomes and broke the camera. `docs/v1/DECISIONS.md` §4.4 |
| **A failed tool must never render like an empty result** | DDG's CAPTCHA is HTTP 202, so a block was reported as "nothing found". `docs/v1/DECISIONS.md` §3.6 |
| **Flag network tools `blocking=True`** | They run on the event loop otherwise and stall every live tick. `docs/v1/DECISIONS.md` §3.7 |
| **Image cost scales with resolution, not JPEG quality** | Quality is not a lever. §5.1 |

---

## Constraints

- **v2 cannot run on Groq's free tier.** A tick is ~1,440 tokens against an
  8,000 TPM cap — one tick per 11s, versus a 4s default interval — and the
  200,000 TPD cap is ~139 ticks *total per day*. This is now enforced rather
  than documented: `LIVE_BACKEND_MODE` defaults to `deepinfra`, and a
  Groq-vision backend raises at startup unless `LIVE_ALLOW_GROQ_VISION=true`.
  §5.2, §5.3
- **DeepInfra Qwen3-VL-30B-A3B is ~$0.26 per 1,000 ticks.** Matching Groq's
  entire free daily allowance costs about four cents. Watch
  `DeepInfra vision usage:` — in this split the vision call is the *only* image
  cost, so its `prompt_tokens` **is** the per-frame bill.
- **Render sleeps after ~15 min** with no *inbound* traffic. `/v2/poll` on a
  20s interval keeps it warm during use.
- **`DEFAULT_TIMEZONE`** must be an IANA name (`Europe/Berlin`, never `CEST`).
  Needs `tzdata` (pinned), or `zoneinfo` falls back to UTC on hosts with no
  system tz database.
- **`SEARCH_EXCLUDED_DOMAINS` defaults to `wikipedia.org`.**

---

## Layout

```
CLAUDE.md          this file — v2 working reference
HANDOFF.md         current status + what to do next
DECISIONS.md       v2 failure log + design reasoning
docs/v1/           the superseded system's reference and failure log
render.yaml
server/
├── main.py                FastAPI app; includes both routers
├── config.py              shared settings
├── live/                  ── v2 ──────────────────────────────────
│   ├── agent.py           LiveAgent — tick/chat/poll, write windows, prompts
│   ├── worlddoc.py        the document: state, mutation, render()
│   ├── tools.py           13 tools
│   ├── triggers.py        the zero-cost arithmetic engine
│   ├── vision.py          tick vision prompts (form / read / questions)
│   ├── compaction.py      span-preserving summarisation
│   ├── config.py          v2 settings, all env-overridable
│   └── routes.py          /v2/* and the /live page
├── backends/
│   ├── deepinfra_backend.py  ACTIVE for v2 — DeepInfra vision + DeepSeek reasoning
│   ├── deepseek_backend.py   ACTIVE for v1 — Groq vision + DeepSeek reasoning
│   └── factory.py            get_backend()
├── agent/                 ── v1, superseded but running ──────────
└── static/
    ├── live.html, live.js        v2 UI
    ├── index.html, app.js        v1 UI
    └── sw.js                     PWA shell — excludes v2 entirely
```

---

## Current state

**v2 has been run against live traffic once** — a ~18-minute chicken-curry
session on 2026-08-10. Propose-then-commit, caption reuse, the urgent path and
the blank-camera report all behaved correctly. The session ended on a **Groq**
rate limit, which v2 should never touch. See `HANDOFF.md`.

Verification is `tests/run_all.py` (13 ad-hoc harnesses) plus reading exported
sessions. There is no automated test suite. The JS harnesses drive the real
`live.js` in a stubbed DOM rather than reimplementing it — note that top-level
`let` is not a property of a `vm` sandbox, so state must be poked via
`runInContext`.

**The export is the highest-value artifact.** It carries every vision prompt
and answer, silent ticks included. Reading it top to bottom is how the
flip-flops were found — no single turn shows them, only the sequence does.
