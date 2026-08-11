# Vision Chitragupta v2 — Design Context

**Purpose of this file.** Self-contained context for continuing design discussion
about this system in a fresh conversation. Paste it in whole. It supersedes
`CLAUDE.md` for discussion purposes — v1 is not described here at all, and v2 is
the only system under development.

Everything below is drawn from the code in `server/live/` as it actually stands,
not from aspiration. Where the code and the intent disagree, that is flagged.

---

## 1. Thesis

> **The world document is primary state. Speech is a side-effect.**

A hands-free, camera-equipped voice assistant for hands-on work — cooking,
repairs, shopping. The user is **listening, not reading**, and their hands and
eyes are busy. That single constraint drives every design choice.

The naive architecture asks, on every frame, *"what should I say about this
frame?"* That architecture narrates. It cannot do otherwise, because a model
asked to produce speech produces speech.

v2 inverts it. Ticks continuously update a persistent document of everything
seen and decided. A separate, **zero-token arithmetic engine** decides when the
user is owed something. The reasoning model is asked "should you speak?" only
as a distinct, separately-prompted question — and most of the time it is not
asked at all.

Three consequences follow, and they are the whole system:

1. **Silence is the default and it is cheap.** An idle tick costs one vision
   call and one reasoning call, and the speech question is skipped entirely.
2. **Memory is explicit and inspectable.** Not a context window — a JSON
   document with named sections, rendered into every prompt.
3. **The decision to speak is decoupled from the ability to see.** Arithmetic
   over timestamps fires the triggers; the model only judges.

---

## 2. Prior art: VideoLLM-online (CVPR 2024)

The closest published system, and the right reference point for this discussion.
Show Lab (NUS) + Meta. [Project page](https://showlab.github.io/videollm-online/) ·
[CVPR open access](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_VideoLLM-online_Online_Video_Large_Language_Model_for_Streaming_Video_CVPR_2024_paper.html) ·
[code](https://github.com/showlab/videollm-online)

**The problem it names is exactly ours.** Standard video LLMs are *offline*:
they consume a complete clip and answer a question about it. A streaming
assistant must instead decide, at every incoming frame, whether this is a moment
that warrants speech at all — and stay quiet otherwise. They call the failure
mode out directly: existing multimodal models treat video as predetermined
clips rather than a continuous stream.

**Their answer — the LIVE framework (Learning-In-Video-strEam)**, three parts:

1. **A training objective** that performs language modeling over continuous
   streaming input, producing *frame-aligned* responses — the model learns, as
   part of training, at which frames to respond and at which to remain silent.
2. **A data generation scheme** converting offline temporal annotations (Ego4D
   narrations) into streaming dialogue format by prompting an LLM. Existing
   datasets have no "when to speak" supervision, so they synthesize it.
3. **An optimized inference pipeline** — 5–10 FPS on an RTX 3090, 10–15 FPS on
   an A100, for a 5-minute Ego4D clip.

**The one-line summary:** VideoLLM-online solves *when to speak* **inside the
model, by training**. Frames and text are interleaved in one temporally-ordered
autoregressive sequence, and silence is a learned output.

---

## 3. What we changed, and why

Our binding constraint is the opposite of theirs: **we train nothing.** Two
independently-trained, off-the-shelf hosted models from two vendors, called over
HTTP. Every divergence below follows from that one fact.

| Dimension | VideoLLM-online | Chitragupta v2 | Why we diverge |
|---|---|---|---|
| **When-to-speak** | Learned. Frame-aligned response emitted by the trained model | **Arithmetic**, outside any model (`triggers.py`), plus a separate LLM speech call | We cannot add a training objective. So the decision moves out of the model into code we control |
| **Vision↔language interface** | Continuous embeddings, interleaved in one sequence | **Plain text captions** | Two vendors, never co-trained, no shared latent space. Text is the only interface that exists between them |
| **Memory** | Transformer KV cache over the whole stream | **Explicit JSON world document**, re-rendered into every prompt | No persistent cache across independent API calls. Also: ours survives restarts and is human-auditable |
| **Frame rate** | 10–15 FPS | **~0.16 FPS** (one tick per ~6s), gated by a perceptual diff | Every frame is a billed API call, not a local forward pass |
| **Cost model** | Watts, on your own GPU | **Tokens**, per frame, per vendor | Drives the entire cost-control layer, which they need none of |
| **Silence** | A learned token | An explicit `[SILENT]` sentinel string, stripped before display | Same function, implemented in prompt-and-parse rather than in weights |
| **History bound** | Context length | `RECENT_MAX=24` captions, overflow **compacted** into narrative by a cheap model call | Cost, not capability. Raw captions are never silently dropped |
| **Temporal grounding** | Learned frame-position encoding | **Readable wall-clock timestamps** on every entry, in the *user's* timezone | The model does temporal arithmetic against text it can literally read |

### The deeper trade

They bought **tight coupling**: low latency, no serialization loss, learned
silence. They paid in **flexibility** — swapping their vision encoder is a
retraining run.

We bought **modularity**: swapping the vision model is a config change, and
every intermediate state is human-readable text we can debug by reading. We pay
in **latency** (two sequential network round-trips) and **lossiness** (the
caption is a bottleneck; whatever it omits is gone forever).

**The bet:** a *human* is the actuator. A human is slow but extraordinarily
smart, which buys a latency budget measured in seconds instead of milliseconds.
At seconds-scale, text is affordable. That is the whole reason this architecture
is viable and a robot's is not.

---

## 4. Architecture

```
Phone/browser ──► /v2/tick   camera frame, on an interval
                  /v2/chat   the user said something
                  /v2/poll   heartbeat — arithmetic only, usually free
                      │
                      ├─► [Vision]    Qwen3-VL-30B-A3B on DeepInfra
                      │               frame + brief ──► plain-text caption
                      │               never reasons, never decides
                      │
                      ├─► [Reasoning] DeepSeek v4-flash — text only, never
                      │               sees pixels. Owns every decision and tool call
                      │
                      ├─► [Triggers]  pure arithmetic over the doc. Zero tokens
                      │
                      └─► world document  (+ optional speech)
```

### The tick, in five phases

`server/live/agent.py::tick`. Only the odd-numbered phases hold the lock.

| Phase | | Lock | ~Cost |
|---|---|---|---|
| 1+2 | caption the frame | **unlocked** | ~1.5 s, one vision call |
| 3a | fold caption in, claim triggers | locked | ~1 ms |
| 3b | **Stage 1** — bookkeeping reasoning, tools only | **unlocked** | ~2 s |
| 3c | is the speech question even worth asking? | locked | ~1 ms |
| 3d | **Stage 2** — the speech decision | **unlocked** | ~1 s |
| 3e | politeness gate, record the utterance | locked | ~1 ms |

**Why two reasoning stages.** Stage 1 is scored on exactly one thing: *is the
document now accurate?* Its prose output is discarded. Stage 2 gets a small,
separate prompt — no tools, no system brief, no full document — answering one
question: *does the user need to hear something?*

A single call scored on both objectives **always chose silence.** That failure
was reported three times before the split fixed it. Bookkeeping and speech are
different jobs with different success criteria, and a model asked to do both
optimizes the one it can satisfy without risk.

**Phase 3c is a cost gate.** `worth_asking` is true only if tools fired, or
triggers fired, or we are inside the follow-up window, or a proposal is pending.
Otherwise 3d is skipped entirely and the tick costs one reasoning call, not two.

### Two yield points — priority preemption

`self._user_waiting` is set by `chat()` **before** it queues, not after it
acquires. A tick in flight checks it twice ([agent.py:701] and [agent.py:742])
and **abandons its own remaining work** if a person has started talking.

The tick keeps the caption it already paid for, writes it, and returns. Its
commentary is disposable; a person waiting is not. This is genuine
priority scheduling, and it is the most real-time-systems-shaped mechanism in
the codebase.

### The lock covers writes, not thinking

`LiveAgent._lock` is held **only** inside `_write_window()`: reload → mutate →
save → release. Milliseconds.

**Every model call happens outside one**, including the reasoning that produces
tool calls — `_reason(deferred_writes=True)` hands them to `_apply`, which takes
its own window.

The cost: a turn reasons about a document that may have moved by the time it
writes. Hence two invariants:

- **Every write window reloads.** Never reuse a doc read before a model call.
- **Doc-mutating tools degrade to a harmless "no match" string** rather than
  raising when their target is gone.

Losing a tick's bookkeeping to a race is recoverable — the next frame re-derives
it. Making the user wait is not.

---

## 5. The world document

`server/data/live/worlddoc.json` — gitignored, survives restarts by design.
`worlddoc.render()` produces the prompt text, in this order:

```
[Current time: HH:MM:SS]
[Goal] <title>
[Tasks]                          [ ] pending  [~] in_progress  [x] completed  [-] skipped
[PROPOSED PLAN — NOT COMMITTED]  only when awaiting the user
[Camera focus — fine frames, N used]
[Open expectations]              time-anchored show a countdown; event-anchored show the watch
[Earlier this session]           compacted narrative, with time spans
[Known environment facts]        durable spatial memory
[Recent observations]            raw captions, newest last
```

**Section order is stability-first.** Title/tasks/narrative/environment change
rarely; `recent` changes every tick. This gives DeepSeek's prefix cache the
longest possible unchanged prefix across consecutive ticks. Ordering is a
cost-and-latency decision, not a readability one.

**Every timestamp is in the user's zone** (`DEFAULT_TIMEZONE`, an IANA name),
not the server's. Render runs UTC on Render.com; a naive `fromtimestamp()` once
stamped the entire document — including the `[Current time]` header the model
does all its temporal arithmetic against — two hours off. "Will this be done by
four?" was answered about a different four.

**`rev` is monotonic**, bumped on write-window *entry* (entry order is write
order, since windows serialize on the lock). Every response echoes it as
`doc_rev`; the client discards any render older than one it has already painted,
because concurrent ticks and chat turns reply out of order.

### Overflow

`recent` is bounded at `RECENT_MAX` (24). On overflow the oldest `COMPACT_BATCH`
(16) captions are summarized into `narrative` by one cheap model call —
**time spans preserved**, not just facts — plus any durable environment facts
worth promoting. Raw captions are never silently dropped, and the freshest
window is never compacted.

Compaction is **the one deliberate exception** to "no model call inside a write
window": it rewrites `recent` and `narrative` together, so it cannot be replayed
against a doc that moved underneath it. It runs once every `RECENT_MAX` frames.

---

## 6. Triggers and the speech gate

`triggers.check(doc)` — pure arithmetic, no tokens:

| Kind | Fires when | Priority |
|---|---|---|
| `expectation_due` | a time-anchored expectation passed its deadline unresolved | inherited |
| `proposal_pending` | a plan proposed and unanswered for `PROPOSAL_RERAISE_S` (150 s) | normal |
| `stale_task` | an `in_progress` task unmentioned for `STALENESS_S` (480 s) | low |

**Each claims its state on firing** — sets `status="fired"` / bumps
`raised_ts` / resets `last_mention_ts` — *before* any awaited model call
downstream. Without the claim, the same event fires on every subsequent tick.

`stale_task` is suppressed when the task has its own open time-anchored
expectation: the deadline will speak for it, and two nags for one thing is worse
than none.

Event-anchored expectations **never appear here** — their condition can only be
judged against a frame, so they ride along in the vision prompt instead.

### The politeness budget

Unprompted speech waits out `MIN_UNPROMPTED_GAP_S` (90 s). Two things bypass it:

- **`[URGENT]`** — physical risk, or work about to be ruined. **One flag, one
  consequence:** it bypasses the gate and does nothing else. It does not change
  capture detail, does not affect which questions are asked, does not resolve
  anything.
- **The follow-up window**, `FOLLOWUP_WINDOW_S` (180 s). Answering the user used
  to *reset* the politeness gap, which silenced exactly the follow-up they were
  waiting for. Asked to find the onions, the assistant said "I'll point them out
  when I see them" — and that reply gagged it for the full 90 s search. It found
  them at +27 s, logged them silently, and said nothing until asked again.

  `last_user_turn_ts` is tracked **separately** from `last_spoken_ts` because
  the two pull in opposite directions: **speaking should make the assistant
  quieter; being asked should make it more forthcoming.** Only the tick path
  opts into the window — `poll()`'s stale-task nags still respect the gap.

---

## 7. The vision stage

The reasoning model cannot see. It aims the camera two ways.

### `set_vision_focus(brief, detail, mode)` — the standing lens

One or two plain sentences saying **what the user is physically doing**.
Replaced, never appended, so duplicates are impossible by construction. (Nine
overlapping form watches once accumulated on a single oil-filter plan, five of
them restatements, because every planning pass could add more and nothing could
merge them.)

- **The model writes only the activity.** The grip/posture/danger instructions
  are attached automatically to every frame by `vision.py`. When the model wrote
  the whole block it wrote *checklists* — "whether a drain pan is directly
  underneath the filter" — which made every different-but-fine setup read back
  as a list of absent items. The reasoning model cannot see the user's garage,
  so it must not be the one describing it.
- **`mode="form"`** → report posture, grip, danger, and anything else out of
  place. Explicitly *"not a checklist"*, and *"absence is not a finding."*
- **`mode="read"`** → transcribe text verbatim, state what is illegible and
  precisely how to fix it (rotate, flatten, tilt from the glare). Forces
  `detail="fine"` and loads **none** of the form wording. Asked to read a
  frozen-meal packet in form mode, the camera reported "hand position is not
  visible" while the cooking instructions went untranscribed.
- **`detail`**: `fine` (1024 px) vs `coarse` (640 px). Bounded by
  `MAX_FINE_FOCUS_FRAMES` (120) — a nudge-then-backstop, because a fine mode set
  once is never voluntarily reverted. The budget carries over only on an exact
  no-op re-assertion; any real change earns a fresh budget.

### `set_expectation(anchor="event", condition=…)` — a discrete watch

The condition is put to the camera **verbatim as a question on every frame**
until resolved. Answers come back **first**, before the description, as:

```
Q1: FOUND — <exactly where, plus any label text>
Q1: NOT VISIBLE — <what is in that part of the frame instead>
Q1: UNCLEAR — <what is legible, and what blocks a confident answer>
```

The question count is stated **twice** in the prompt on purpose: asked a single
question, the model answered "Q1/Q2/Q3", inventing two more to hang its
observations on.

`MAX_ACTIVE_BRIEFS` (4) reach any one frame, high-priority first — the vision
model cannot answer nine questions and describe a scene in one reply.

### The load-bearing rule

> **A brief must ask for observations, never judgement.**

"Is the grip safe?" returns a reassuring guess. "Are the fingertips curled back
or extended flat?" returns a fact. The prompt says outright: *never say
something looks safe, correct, careful or fine — a wrong reassurance is the most
damaging answer you can give.* The reasoning stage decides; the camera reports.

Without explicit questions this failed silently: the user wanted black-eyed
beans, the caption said "several bags of lentils", and the reasoning model —
with no answer to read — upgraded that into "I can see the beans." An inference
stood in for an observation because no question was posed.

### Two more prompt-level rules

- **Each tick is given the previous caption as text**, so it describes *change*
  rather than re-describing the scene. This is the hosted-API substitute for a
  real temporal encoding: the comparison baseline arrives in words.
- **Never describe the camera.** A whole session of captions once read "the
  camera pulls back and tilts upward", "the camera pushes in close" — the model
  narrating the videographer instead of the kitchen. The user is holding the
  phone and knows they moved it. Full price, zero information.

---

## 8. Plans are proposed, not written

The split is **what it saw** vs **what it decided**.

**Observations** — captions, `log_environment`, `resolve_expectation`, focus
changes — write **silently and immediately**. Gating them would turn every tick
into a permission prompt.

**A plan** is a decision about how the user spends the next hour, made from a
photograph and a web search. Once it lands in `tasks` it is re-injected into
every later prompt as settled fact, and the model reads its own guess back as
memory and holds the user to it.

```
propose_plan  → doc["proposal"], NOT tasks. Nothing tracks it, no expectations.
                MUST be said out loud in the same reply — the user cannot see it.
commit_plan   → promotes it to real tasks. Called on assent, including implicit
                ("yes", "go on", or visibly starting step one).
discard_plan  → drops it.
```

Rendered under a hard `[PROPOSED PLAN — NOT COMMITTED]` banner with explicit
instructions not to act on it. A proposal that reads like a task list is worse
than no proposal at all.

**A tick may only commit on visibly starting step one.** A frame cannot tell you
someone said yes.

---

## 9. Tools (13)

Doc-mutating tools close over the agent's current in-memory doc, so a turn's
tool calls and the agent's own writes can never interleave on disk.

| Group | Tools |
|---|---|
| Approval cycle | `propose_plan` · `commit_plan` · `discard_plan` |
| A plan already underway | `update_tasks` · `mark_task` |
| Deadlines and watches | `set_expectation` · `resolve_expectation` |
| The standing lens | `set_vision_focus` |
| Durable spatial memory | `log_environment` · `retract_environment_fact` |
| Inherited, unchanged | `web_search` · `fetch_page` · `calculate` |

**Deliberately absent:** `start_timer` (subsumed by a time-anchored expectation,
which also has a resolution path timers never had) and `request_camera` /
`request_live_search` (the live UI owns the camera; chat turns attach the current
frame client-side).

**`retract_environment_fact` takes a `correction`, not just a deletion.** The
raw captions that produced the wrong inference are still in `recent` and will
suggest it again on the very next tick. A hole in the fact list does not block
that; a durable *"the bag on the pantry shelf is NOT toor dal"* does. A false
fact that survives an explicit correction is worse than no fact at all, because
once logged it is indistinguishable from a verified one.

---

## 10. Client-side cost control (`live.js`)

| Constant | Value | Role |
|---|---|---|
| `FRAME_DIM` | `{coarse: 640, fine: 1024}` | caps the **longest** side |
| `JPEG_QUALITY` | 0.85 | cost scales with resolution, not quality — quality is not a lever |
| `CAPTION_REUSE_MS` | 15000 | a chat turn reuses the last caption if the scene has not moved |
| `FLAT_FRAME_STDDEV` | 2.0 | below this the frame is blank, not merely unchanged |
| `POLL_INTERVAL_MS` | 20000 | keeps Render warm (sleeps after ~15 min without inbound traffic) |

**The diff gate** (32×32 grayscale, mean absolute delta) is the main cost
control: if the scene has not meaningfully changed, no request leaves the
browser.

**Blank-frame detection runs before the diff gate.** A black frame *is* an
unchanged frame to a delta comparison, so a dead camera reported "nothing
changed" indefinitely. Standard deviation is the liveness test.

The server echoes `frame_detail` on every response and the client sizes its
**next** capture from it — resolution discarded in the browser cannot be
recovered, so the decision must run one frame ahead.

**`tickBusy` and `chatBusy` are separate flags with separate queues.**

---

## 11. Hard rules

Each cost a real debugging session.

| Rule | Why |
|---|---|
| **Never `await` a model call inside a write window** | It holds `_lock` and re-serializes ticks against chat, undoing the phase split. `compaction.compact` is the one deliberate exception |
| **Every write window reloads** | Reusing a doc read before a model call silently rolls back whoever wrote in the meantime |
| **Never gate a chat send on `tickBusy`** | One shared `busy` flag kept the browser from sending a question until the tick returned, defeating every server-side overlap |
| **Apply a doc render only if `doc_rev` is newer** | Concurrent turns reply out of order; a slow tick's render predates a chat's writes and will stamp over it |
| **A brief must ask for observations, not judgement** | "Is the grip safe?" gets a reassuring guess |
| **Never add a pre-filter whose "no" looks like silence** | Cost a whole class of silently dropped frames |
| **One flag, one consequence** | A single flag once drove three unrelated outcomes and broke the camera |
| **A failed tool must never render like an empty result** | DDG's CAPTCHA is HTTP 202, so a block was reported as "nothing found" |
| **Flag network tools `blocking=True`** | They run on the event loop otherwise and stall every live tick |
| **Image cost scales with resolution, not JPEG quality** | Quality is not a lever |
| **Never store a countdown; store an absolute deadline** | Restart resilience |
| **Recording something is never the same as answering someone** | In `SYSTEM_BRIEF` verbatim. Silent bookkeeping is correct on idle ticks and a failure when someone is waiting |

---

## 12. Constants

`server/live/config.py`, all env-overridable with a `LIVE_` prefix.

| Constant | Default | Meaning |
|---|---|---|
| `RECENT_MAX` | 24 | raw captions kept verbatim |
| `COMPACT_BATCH` | 16 | oldest captions consumed per compaction |
| `MAX_ENV_FACTS` | 30 | durable facts (oldest dropped) |
| `MAX_NARRATIVE` | 20 | compacted narrative entries |
| `STALENESS_S` | 480 | in-progress task with no mention → check-in |
| `MIN_UNPROMPTED_GAP_S` | 90 | politeness budget |
| `FOLLOWUP_WINDOW_S` | 180 | window after a user turn where the gap is waived |
| `MAX_BRIEF_ASKS` | 40 | asks before nudging the model to resolve a watch (~4 min) |
| `PROPOSAL_RERAISE_S` | 150 | unanswered plan re-raise |
| `MAX_ACTIVE_BRIEFS` | 4 | watches per frame |
| `MAX_FINE_FOCUS_FRAMES` | 120 | fine-detail backstop (~12 min) |
| `VISION_MAX_TOKENS` | 200 | base vision reply budget |
| `VISION_TOKENS_PER_QUESTION` | 60 | added per active watch |
| `TICK_MIN_INTERVAL_S` | 1.5 | server-side floor between accepted ticks |

### Operational constraints

- **v2 cannot run on Groq's free tier.** A tick is ~1,440 tokens against an
  8,000 TPM cap — one tick per 11 s versus a ~4 s interval — and the 200,000 TPD
  cap is ~139 ticks *total per day*. `LIVE_BACKEND_MODE=deepinfra` is required.
- ✅ **`LIVE_BACKEND_MODE` now defaults to `deepinfra`, and a Groq-vision
  backend is refused at startup.** Previously it defaulted to `"hybrid"` — the
  Groq path — so the default nobody sets was the one configuration that cannot
  work, and it failed *late*. Three changes closed it:
  - the default is `deepinfra`;
  - `VisionBackend.VISION_PROVIDER` declares where the pixels actually go,
    because it was not otherwise knowable — `DeepInfraHybridBackend` extends
    `DeepSeekBackend` and swaps out a vision client built in the parent's
    `__init__`, so neither the class name nor the mode string tells you;
  - `_check_vision_provider()` raises on `"groq"` unless
    `LIVE_ALLOW_GROQ_VISION=true`, and `main.py` warms the agent at startup so
    the check runs at boot rather than on the first tick. It does **not**
    re-raise there: v1 shares the process and must keep serving.

  The startup line to check is now
  `Initialized live agent | backend mode: … | VISION ON: … | reasoning: …`
  (ASCII on purpose — a Windows cp1252 console raises `UnicodeEncodeError` on
  an em dash, so a diagnostic containing one dies while reporting the problem
  it exists to explain).
- **DeepInfra Qwen3-VL-30B-A3B is ~$0.26 per 1,000 ticks.** In this split the
  vision call is the *only* image cost, so its `prompt_tokens` **is** the
  per-frame bill.
- **`DEFAULT_TIMEZONE`** must be an IANA name (`Europe/Berlin`, never `CEST`),
  and needs `tzdata` pinned or `zoneinfo` silently falls back to UTC.
- **Camera/mic need HTTPS or the literal hostname `localhost`.** A bare LAN IP
  over HTTP fails the browser's secure-context check.
- **Avoid `--reload`** — WatchFiles has served stale bytecode on Windows.

---

## 13. Layout

```
server/
├── main.py                    FastAPI app
├── config.py                  shared settings
├── live/                      ── v2 ──
│   ├── agent.py     (1021)    LiveAgent — tick/chat/poll, write windows, prompts
│   ├── worlddoc.py   (592)    the document: state, mutation, render()
│   ├── tools.py      (346)    13 tools
│   ├── vision.py     (237)    tick vision prompts (form / read / questions)
│   ├── triggers.py   (127)    the zero-cost arithmetic engine
│   ├── routes.py     (104)    /v2/* and the /live page
│   ├── compaction.py  (74)    span-preserving summarisation
│   └── config.py      (75)    v2 settings
├── backends/
│   └── deepinfra_backend.py   ACTIVE — DeepInfra vision + DeepSeek reasoning
└── static/
    └── live.html, live.js     v2 UI
```

Run: `uvicorn server.main:app --host 0.0.0.0 --port 8000`, then
`http://localhost:8000/live`. Requires `TOOLS_ENABLED=true` and
`LIVE_BACKEND_MODE=deepinfra`.

---

## 14. Current state

**Run against live traffic once** — a ~18-minute chicken-curry session,
2026-08-10. Propose-then-commit, caption reuse, the urgent path and the
blank-camera report all behaved correctly. The session ended on a **Groq** rate
limit, which v2 should never touch — now fixed and guarded at startup (§12).

Verification is `tests/run_all.py` (13 ad-hoc harnesses) plus reading exported
sessions. **There is no automated test suite.** JS harnesses drive the real
`live.js` in a stubbed DOM rather than reimplementing it — note that top-level
`let` is not a property of a `vm` sandbox, so state must be poked via
`runInContext`.

**The export is the highest-value artifact.** It carries every vision prompt and
answer, silent ticks included. Reading it top to bottom is how the flip-flops
were found — no single turn shows them, only the sequence does.

---

## 15. Open thread: making it feel real-time

Active discussion. The reframe that organizes it:

> A robot's deadline is set by physics. Ours is set by human conversational
> tolerance. The metric is **time-to-first-spoken-syllable (TTFS)**, not
> tokens/sec and not end-to-end completion.

`<500 ms` feels instant · `~1 s` responsive · `2–3 s` laggy, users repeat
themselves · `>4 s` broken.

### Three distinct kinds of "parallel" — two are done

1. **Mutual exclusion removed — DONE.** `_vision_for()` captions with the lock
   released, so a chat turn's reasoning genuinely overlaps a tick's captioning.
2. **Preemption — DONE.** The two `_user_waiting` yield points.
3. **Data dependency broken — NOT DONE.** Within one tick, phase 3b cannot start
   before phase 1+2 finishes, because the prompt is built from the caption.

**Lock-free is not pipelined.** On an idle session with no user talking there is
only one workflow, so the three unlocked phases run strictly serially:
~1.5 s + ~2 s + ~1 s ≈ 4.5 s per tick, with nothing to overlap against. The lock
removal pays off only under contention, and contention is the exception.

### Ranked candidate work

1. **Sentence-chunked TTS.** Currently the client speaks only the *finished*
   answer — the streaming endpoint feeds the screen, not the ear, while the
   stated design constraint is that the user is listening and not reading.
   Buffering deltas to a sentence boundary and speaking each sentence as it
   completes is the direct analogue of **action chunking** in robot policies:
   emit a chunk, execute it while the next is computed. Plausibly 3–5 s → under
   1 s of TTFS, no server change, no extra tokens. Constraints: `[SILENT]`
   arrives as the first tokens and must be caught before anything is spoken;
   don't speak a preamble from a pass that ends in a tool call; `synth.cancel()`
   on barge-in.
2. **Prefix-cache-aware prompt ordering** — already done in `worlddoc.render()`;
   worth auditing the tick and chat prompt *wrappers* around it for volatile
   text sitting above stable text.
3. **A vision watchdog.** If the caption has not returned in N ms, proceed on the
   previous one marked stale. *Degrade, never block.*
4. **Software pipelining across ticks.** Let tick N's reasoning consume tick
   N−1's caption, so vision and reasoning are in flight simultaneously even with
   no user present: steady-state cost `vision + reason` → `max(vision, reason)`.
   **Heavier than it looks** — the caption is not just prompt text, it also
   drives `add_recent()` ordering and the compaction batch, and it breaks the
   invariant that the prompt's caption describes the frame just captured.
5. **Adaptive tick rate.** Rate should track scene volatility. The diff gate
   already measures volatility and that signal is unused for pacing.
6. **Pre-authorized decisions (ambitious).** Today triggers only *wake* the
   model. Let the model *pre-commit* — "for the next 3 ticks stay silent unless
   the pan smokes or the timer fires" — and let the arithmetic engine execute
   that at zero cost. This is real action chunking rather than an analogy, and
   it amortizes a reasoning call across several ticks.

### Explicitly not worth doing

Quantization, compilation, RTOS/jitter work, distillation. The system is
**network-bound, not compute-bound** — none of the latency is arithmetic we
control. The only available levers are *overlap it, hide it, or skip it*, which
is why every item above is about scheduling and none is about making anything
faster.

### Blocking gap

**There is no wall-clock instrumentation.** Token usage is logged per stage;
elapsed time is not. Nobody currently knows whether vision is 400 ms or 2.5 s,
or what share of TTFS is generation versus the TTS engine's own startup. One
timing log per phase turns this list from plausible into ranked.
