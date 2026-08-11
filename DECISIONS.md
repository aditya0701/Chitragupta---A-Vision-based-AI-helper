# Decisions — v2

The failure log and the reasoning behind the live tick system (`server/live/`,
`/live`, `/v2/*`). **Read the relevant section before changing an area.** Most
of the non-obvious code here is scar tissue from a specific bug, and the
comment in the source usually points back to a numbered section below.

**v1's log is archived at `docs/v1/DECISIONS.md`.** Code comments that say
`DECISIONS.md §6.3` or `DECISIONS.md 4.4` mean *that* file — those numbers were
assigned before this one existed and are not renumbered here.

---

## 1. Why v2 exists

### 1.1 The inversion

v1 asked, on every frame: *what do I say about this?* That question has an
answer every time, which is why it narrated. Silence had to be bolted on
afterwards as a `[SILENT]` protocol, a regex detector for leaked
silence-narration, and a list of near-variants — three mechanisms fighting the
prompt's own framing.

v2 asks a different question. **The world document is primary state; speech is
a side-effect.** Ticks update the document continuously and silently. A
separate engine decides, independently, when the user is owed something. Most
ticks correctly produce no speech because producing speech was never the
objective.

The tick prompt says this outright: *"Your ONLY job on this tick is to make the
document match what is now true. Speaking is decided separately, afterwards, by
another step — so do not weigh whether to say anything."*

### 1.2 Why the trigger engine is arithmetic

Wall-clock math is free; tokens are not. This is v1's `timers.py` lesson
generalised to the whole system. `triggers.check()` runs on every tick and
every 20s poll, costs nothing, and the reasoning model is only woken when it
returns something — or when a frame arrived, or the user spoke.

Each trigger **claims** its state on firing (`exp["status"] = "fired"`,
`proposal["raised_ts"] = now`) *before* any awaited model call downstream. Same
double-fire lesson as v1's `timers.mark_firing()`: an unclaimed trigger fires
again on the next tick while the first one is still thinking.

Event-anchored expectations never appear as trigger events. Their firing
condition can only be judged against the current frame, so they ride along in
the rendered doc and go to the camera as questions instead.

---

## 2. Concurrency: the lock covers writes, not thinking

The world document is one JSON file, so `load → mutate → save` must be atomic
and there is one `asyncio.Lock`. The question was only ever *how long to hold
it*.

Originally: across the whole tick. That meant a tick owned the document for two
DeepSeek round trips plus any `web_search` they made — **2 to 8 seconds** —
during which nothing was being written and a person asking a question simply
queued. The write itself is a millisecond.

Now it is held only inside `_write_window()`. Measured with
`tests/t_live_parallel.py`, at the point where a question arrives mid-reasoning:

| | time to reach the wire |
|---|---|
| before | 2.00s |
| after | 0.01s |

### 2.1 Never `await` a model call inside a write window

This is the rule the whole restructure exists to enforce. A network call inside
a window re-serializes ticks and chat and silently undoes the phase split — the
code still looks correct, it is just slow again.

**`compaction.compact` is the one deliberate exception.** It rewrites `recent`
and `narrative` *together*, so it cannot be replayed against a document that
moved underneath it, and it runs once every `RECENT_MAX` frames rather than
every tick. The cost is bounded and the alternative is a torn history.

### 2.2 Every window reloads

The point of releasing the lock across model calls is that someone else may
write in the meantime. A window that reuses a doc read before a model call
silently rolls their write back — a lost-update bug that leaves no trace,
because both writes "succeeded".

So `_write_window()` calls `worlddoc.load()` on entry, unconditionally. The
caption computed before the window is still valid: it describes a *frame*, not
the document.

### 2.3 The cost, accepted deliberately

A turn now reasons about a document that may have moved by the time it writes.
That is real, and it is why the doc-mutating tools all degrade to a harmless
`"No task matching 'x'"` string rather than raising when the thing they name is
gone.

Losing a tick's bookkeeping to a race is recoverable — the next frame
re-derives it. Making the user wait is not.

### 2.4 The yield points

`chat()` sets `self._user_waiting = True` **before queueing for the lock**, not
after acquiring it. The whole point is for a tick that already holds it to
notice.

`tick()` checks it twice:

- **Phase 3a**, before the bookkeeping call — keeps the caption (already paid
  for, and the freshest thing known) and returns `yielded: True`.
- **Phase 3c**, before the speech decision — the user's own turn is about to
  answer them with more context than this tick has, so don't spend a call
  racing it.

A tick's commentary is disposable; a person waiting is not. Same "latest
matters" logic as `pendingFrame` on the client.

### 2.5 `rev`, and why it's bumped on entry

Concurrency introduced out-of-order replies: a tick that started first can
finish *after* a chat turn that started later, carrying a render from before
that chat's writes.

`doc["rev"]` is bumped on write-window **entry**. Windows are serialized by the
lock, so entry order *is* write order, and a render taken inside the window
carries the rev it will be saved under. Bumping it in `save()` instead would
hand two concurrent responses the same rev and the client could not order them.

### 2.6 Proposed: pipeline the two stages across frames

**Not built. This is a design note, written while the reasoning was fresh.**

The proposal, in its original form: run the two models *in parallel* against a
shared world document, and lock only for writes — borrowing the read/write
concurrency discipline used in compilers and chip design, where reads are free
and only stores are ordered.

Two separate claims are buried in that, and they have different answers.

**The parallelism cannot live inside one frame.** `tick()` awaits
`_vision_for()` and then awaits `_reason()`, and there is no `asyncio.gather`,
`create_task` or `TaskGroup` anywhere in `server/live/`. That is not an
oversight — the reasoning stage's *input is* the vision stage's output. The
caption is the only thing the reasoning model will ever know about the frame,
so the dependency is real and no amount of restructuring removes it.

**It can live across frames.** At steady state, vision captions frame N+1 while
reasoning digests frame N. That is software pipelining, and it is what would
make "two models running at once" literally true. Throughput goes from
`vision + reasoning` to `max(vision, reasoning)` — roughly 3.5s down to 2s.

The concrete argument for doing it: the interval slider is `min="2"`
([live.html](../server/static/live.html)) but an idle tick costs ~1.5s of vision
plus ~2s of reasoning. **The fast half of our own slider cannot keep up**, and
`live.js` schedules the next tick only after the current one returns. Pipelining
is what would make the 2s setting mean anything.

#### The locking half is not needed, and that is the useful finding

The lock is held for three windows of ~1ms inside a ~3.5s tick, and the measured
contention cost is the 0.01s in the table above. Making reads lock-free — an
immutable snapshot with pointer-swap on write, i.e. RCU — is real work that
would measure as approximately zero.

More to the point, **the existing lock already handles this shape of
concurrency**, because it was built for it. Tick-overlapping-chat and
tick-N-overlapping-tick-N+1 are the same problem, and the machinery is already
in place: every window reloads on entry (§2.2), `rev` is bumped on entry so
writes are orderable (§2.5), and the doc-mutating tools already degrade to a
harmless "no match" rather than raising (§2.3). That last one is exactly the
retry-tolerant semantics an optimistic scheme needs; it exists already, for a
different reason.

So pipelining requires **no change to the locking model at all**. It requires
the client to stop serializing whole ticks.

#### What it would actually cost

Tick N's tool writes could interleave with tick N+1's caption fold-in, a race
that cannot happen today because ticks do not overlap.

Most of that is harmless by construction, because the document already has an
ownership split — it is the same "what it saw vs what it decided" line §3 is
built on:

| | owner |
|---|---|
| `recent`, `wanted[].found` | the camera. Only `fold_wanted()` sets `found`, §8.1 |
| `tasks`, `proposal`, `expectations`, `focus`, `environment` | the reasoning model |

Two writers on disjoint sections do not conflict. **The one genuinely shared
section is `wanted`**: the camera sets `found` while the model may `add_wanted`
or `drop_wanted` on the same list. Note that matching is already by name and
never by index for a closely related reason (§8.3) — the failure mode there was
a list rebuilt underneath a positional read, which is the same hazard one turn
earlier.

The risk the harnesses cannot cover is the usual one: they can prove the writes
land, and they cannot tell us whether the *model* behaves when the document
moves under it mid-turn. That needs a live run.

#### If it gets built

- `live.js` only: let a frame be captured and captioned while the previous
  tick's reasoning is still in flight. Keep a hard cap on how many ticks may be
  in flight — pipelining with no bound is just a queue that grows during a
  simmer, and §7.1's `pendingFrame` already establishes "latest matters".
- Nothing in `agent.py` or the lock changes.
- A timed harness alongside `t_live_parallel.py`, driving tick N+1's capture
  into tick N's reasoning window.
- Rollback is one flag on the client, since the server is unchanged.

---

## 3. Plans are proposed, not written

The model writes two categorically different things and they need different
rules.

**Observations** — a caption, `log_environment`, `resolve_expectation`, a focus
change — are reports of what it saw. They write silently and immediately.
Gating them on approval would turn every tick into a permission prompt and
destroy the point of a hands-free assistant.

**A plan** is a decision about what the user is going to spend the next hour
doing, made from a photograph and a web search. Once it lands in `tasks` it is
re-injected into every later prompt as settled fact — so the model reads its
own guess back as memory and holds the user to it. That is the same "a wrong
record survives correction" failure `retract_environment_fact` exists for,
except a plan is far harder to unpick, because tasks, expectations and vision
focus all get built on top of it.

So: **propose out loud, commit on assent.**

- `propose_plan` writes `doc["proposal"]`, never `tasks`. Its return string
  tells the model, in the tool result itself, that nothing is tracking it.
- `render()` emits it as `[PROPOSED PLAN — NOT COMMITTED]` immediately after
  `[Tasks]`, hard-labelled with "the user has NOT agreed" and all three exits.
  A proposal that reads like a task list is worse than no proposal at all.
- `commit_plan` promotes it. Assent includes the implicit kind — "yes", "go
  on", or visibly starting step one.
- One proposal at a time, replaced rather than appended. A second proposal
  means the model changed its mind, not that there are two plans. Same
  reasoning as `set_vision_focus`.

**Expectations belong to a committed plan, not a proposed one.** Attaching
deadlines to steps nobody agreed to means they go overdue while the user is
still deciding.

### 3.1 An unanswered proposal must re-raise

This is the one silence that costs the user something rather than sparing them:
the assistant is blocked on them, they have no idea it is waiting, and nothing
is being tracked in the meantime. `PROPOSAL_RERAISE_S` is 150s — long enough
not to nag someone who is simply thinking.

Anything that actually *asks* — a chat turn calling `propose_plan`, a tick
speaking while a proposal is pending — calls `mark_proposal_raised()`, so the
timer measures from the last time the user was really asked, not from when the
plan was made.

### 3.2 A tick may only commit on visibly starting step one

A frame cannot tell you someone said yes. The tick prompt is explicit:
*"Nothing else on a tick may commit one… If in doubt, leave it pending; they
will be asked."*

### 3.3 Confirmed live

2026-08-10. The assistant proposed a 6-step curry, held it across several
minutes and multiple ticks while the user was doing something else, mentioned
it unprompted (*"I'm still waiting to hear if that style works for you"*), and
called `commit_plan` on "yes this style works for me". Working as designed.

---

## 4. The vision stage

The reasoning model never sees pixels. Everything below is about making a
second model look on its behalf and report something a reasoner can act on.

### 4.1 Ask for observations, never judgement

The load-bearing rule, and it needs restating for bodies specifically. A model
asked *"is the grip safe?"* reaches for reassurance far more readily than one
asked about a jar. A wrong reassurance is the most damaging thing this stage
can produce.

So every brief asks for physical arrangement — *"which fingers are where,
curled or extended, what is in contact with what, which way a handle points"* —
and the prompt says outright: *"Do NOT say whether any of it is safe, correct,
proper, careful or fine — that judgement belongs to the reader."*

The reasoning stage judges. The camera reports.

### 4.2 The reasoning model must not describe the setup

It used to write the whole vision block, and it wrote checklists: *"whether a
wheel chock sits behind a rear wheel; whether a drain pan is directly
underneath the filter"*. That enumerates **one imagined arrangement**, so a
different but perfectly fine setup reads back as a series of absent items —
false findings about a job being done correctly.

The reasoning model cannot see the user's kitchen or garage, so it must not be
the one describing them. `set_vision_focus` now takes **only the activity**
("User is dicing onions on a board at the counter"). The grip/posture/danger
instructions are standard text attached to every frame, identical for every
task.

The prompt also carries the counter-instruction that makes this survive contact
with a real frame: *"This is NOT a checklist and there is nothing to tick off.
If something mentioned above is simply not in this frame, do not mention it at
all — absence is not a finding."*

### 4.3 Read mode is a different job, not a variation

Asked to read a frozen-meal packet, the camera reported *"POSTURE AND GRIP —
hand position is not visible"* while the cooking instructions the user was
holding up went untranscribed. The form block wasn't merely unhelpful, it was
answering a different question.

`mode="read"` loads none of the form wording — different instructions
entirely: transcribe verbatim, numbers and units exactly as shown, keep the
structure, name the language, and where something is illegible say *why* (glare,
angle, folded, too small) and *what would fix it* (rotate, closer, flatten).
The user is holding the thing and can act immediately, but only if told.

Two consequences fall out:

- **Read mode forces `fine`.** Reading is impossible at 640px — that is the
  entire reason the tier exists. Letting `mode="read", detail="coarse"` through
  would silently return "text is not legible" forever, so the mode wins over
  the argument.
- It is ~40% shorter than the form block, so it is cheaper as well as better.

*"Never guess at a character or a digit to complete a word or number"* — a
plausible invention is undetectable downstream.

### 4.4 Never describe the camera

An entire session of captions read *"the camera pulls back and tilts upward"*,
*"the camera pushes in close"*, *"the camera angle shifts to the right"* — the
model narrating the videographer instead of the kitchen.

It is an honest failure: the user is holding the phone and walking around, so
the largest frame-to-frame delta genuinely **is** camera motion, and the
change-framing in §4.5 points straight at it. Those captions cost full price
and carry zero task information.

Fixed by an explicit ban plus a redirect — *"If the view moved to a new place,
name what is now visible there ('pantry shelf: onions in a wooden bowl'), never
the motion that got you there"* — and by teaching the change instruction that a
different viewpoint is not a change.

### 4.5 The previous caption goes back in as text

Each tick is given the previous caption so it describes *change* rather than
re-describing the scene from scratch. This is the hosted-API substitute for
real temporal encodings: the comparison baseline arrives in words.

### 4.6 Watches are asked first, and a "no" is a result

Without an explicit question the brief arrived only as a soft "these are
relevant, be detailed" nudge at the end, so nothing was ever really asked and
nothing sharp came back. Observed live: the user wanted black-eyed beans, the
caption said *"several bags of lentils"*, and the reasoning model — with no
answer to read — upgraded that into *"I can see the beans"*. **An inference
stood in for an observation because no question was posed.**

Now watches are asked first, before the description, one line each, in a fixed
format:

```
Q<n>: FOUND — <exactly where, plus any label text>
Q<n>: NOT VISIBLE — <what is in that part of the frame instead>
Q<n>: UNCLEAR — <what you can make out, and what is blocking a confident answer>
```

`NOT VISIBLE` is a real, useful answer. This is v1's "never add a pre-filter
whose 'no' looks like silence" applied to briefs.

**The count is stated twice on purpose.** Asked with a single question, the
model answered "Q1/Q2/Q3", inventing two more to hang the rest of its
observations on — harmless to read, but it makes the block unparseable and pads
every caption.

### 4.7 Only relevant watches, hard-capped

A full plan registers many watches — a real oil-filter turn produced nine,
from chocking the wheels to seating the new gasket. Sending all of them every
tick is wrong twice: a permanent per-tick token tax, and the vision model
cannot answer nine questions *and* describe the scene inside one reply, so the
answers truncate.

Only watches relevant **now** are asked: tied to an in-progress task, or tied
to no task, or high priority (safety must never wait its turn). Sorted
high-priority-first, then capped at `MAX_ACTIVE_BRIEFS` (4).

The reply budget scales with the ask: `VISION_MAX_TOKENS` (200) +
`VISION_TOKENS_PER_QUESTION` (60) each, or the answers truncate mid-block and
the description never arrives.

### 4.8 No duplicate detection — a Jaccard check was tried and removed

It failed in both directions. The duplicates that actually occurred were
rephrasings sharing only ~19% of their words (*"Car is safe to work under"* vs
*"The car should be settled firmly on both ramps"*), well under any usable
threshold — while a threshold low enough to catch those merged genuinely
distinct watches differing only in the object (*"is the 10mm socket seated"* vs
*"the 12mm"*).

Silently dropping a watch the user is relying on is far worse than carrying a
duplicate. The pressure it was built for is gone anyway: form and safety moved
to `set_vision_focus`, which replaces rather than appends.

### 4.9 The fine budget is a nudge, then a backstop

v1's documented failure was that a fine mode set once is never voluntarily
reverted. v2 keeps the model responsible — it is told, repeatedly, to revert
when the close work is done — and adds two pressures rather than a cutoff:

- `[Camera focus — fine frames, N used]` is rendered into the doc, so the drift
  is **visible** to the thing causing it.
- `MAX_FINE_FOCUS_FRAMES` (120) forces coarse. Generous, because a hands-on
  step legitimately runs a long time — ~12 minutes at a 6s tick.

The frame counter carries over **only** when the focus is re-sent completely
unchanged (a no-op re-assertion). Any real change is new intent and earns a
fresh budget, so a step that legitimately needs another close look is never
locked out by an earlier one — while repeating yourself cannot dodge the cap.

Watches use the same shape: past `MAX_BRIEF_ASKS` (40) a watch **stays open**
but stops buying full resolution, and the tick prompt gets a "going stale"
nudge to close it. A search that has not converged in forty frames will not
converge on the forty-first; dropping it entirely would silently abandon
something the user may still be waiting on.

---

## 5. Cost and providers

### 5.1 Resolution is the only lever

Image cost scales with resolution, not JPEG quality. Quality is fixed at 0.85
and is not a knob.

`FRAME_DIM` caps the **longest** side: coarse 640, fine 1024. A 1024px frame is
~1,000 image tokens on its own.

The `frame_detail` decision has to run **one frame ahead** — resolution
discarded in the browser cannot be recovered, and nothing server-side can
upscale a label back. So the frame that *causes* an upgrade is itself coarse.
The brief outlives it by many ticks, so the answer is not lost.

### 5.2 Groq's free tier cannot run v2 at all

A v2 vision call is ~1,350 input + ~90 output tokens. Groq caps
`qwen3.6-27b` at 8,000 tokens/minute:

```
8,000 TPM / ~1,440 tok per tick  =  5.6 ticks/min  =  one tick per ~11s
```

The default interval is 4s (15 ticks/min) — about 3× over the per-minute cap,
taking 429s inside the first minute. The daily cap is worse: 200,000 TPD /
1,440 ≈ **139 ticks total per day**, which even at the slowest slider setting
is about 35 minutes of watching, once, per day.

v1 survives on Groq because its cadence is slower and most turns are text-only.
v2 ticks continuously **by design** — it is a fundamentally heavier vision
consumer and needs a provider with no per-minute ceiling.

Hence `LIVE_BACKEND_MODE=deepinfra`: Qwen3-VL-30B-A3B at ~$0.26 per 1,000
ticks. Matching Groq's entire free daily allowance costs about four cents. The
seam exists so v1 keeps running on Groq's free tier untouched.

### 5.3 The cook test died on Groq anyway — fixed by making it impossible

2026-08-10, ~18 minutes in, the session ended with:

```
Error code: 429 — Rate limit reached for model `qwen/qwen3.6-27b` …
tokens per day (TPD): Limit 200000, Used 199109
```

**v2 should never touch Groq.** Both `server/.env` (unchanged since 31 July)
and `render.yaml` set `LIVE_BACKEND_MODE=deepinfra`, and
`factory.get_backend("deepinfra")` routes cleanly to
`DeepInfraHybridBackend`, which overrides `vision()` onto DeepInfra's client.
A missing `DEEPINFRA_API_KEY` cannot explain it either — that raises at
construction and would fail *every* request, not produce a Groq 429.

The only path from `/v2/chat` to Groq is `self.backend.vision()` on an
un-overridden `DeepSeekBackend`, i.e. the live agent resolved to `hybrid`. So
the running process did not have `LIVE_BACKEND_MODE` applied — a stale server
predating the config, or an environment where it wasn't loaded.

**The interesting part is that nothing was wrong on disk.** Every file said
`deepinfra`, so there was no artifact anyone could inspect and find a mistake
in. That is the actual defect: the system had a hard requirement and no point
at which it compared that requirement against reality.

Three changes, in order of how much each carries:

**1. The default was the bug.** `LIVE_BACKEND_MODE` defaulted to `"hybrid"` —
the Groq path — a leftover from before the DeepInfra backend existed. So the
one configuration that *cannot work* was what you got by not choosing. A
default nobody sets must be the one that works; it is now `deepinfra`.

**2. "Which provider gets the pixels" was not knowable.** No code could have
checked this even if it had wanted to. `DeepInfraHybridBackend` extends
`DeepSeekBackend` and *replaces* a Groq vision client constructed in the
parent's `__init__` — so the class name, the mode string and the module name
all fail to answer the question. `VisionBackend.VISION_PROVIDER` now declares
it, in the same idiom as `SPLIT_VISION_REASONING` and `SUPPORTS_NATIVE_TOOLS`.

**3. v2 refuses to start on Groq vision.** `_check_vision_provider()` raises
unless `LIVE_ALLOW_GROQ_VISION=true`. §5.2 means the configuration cannot work,
so accepting it only chooses *when* the user finds out, and eighteen minutes
into someone's cooking is the worst available answer.

Three details that are deliberate:

- **The escape hatch is per-run and explicit.** Choosing Groq for a one-off
  comparison is legitimate; *arriving* there is the bug. The distinction the
  check enforces is intent, not provider.
- **`main.py` warms the agent at startup so the check runs at boot — but does
  not re-raise.** v1 shares the process and is the system actually in daily
  use. Taking the whole server down over v2's config would make the fix worse
  than the bug. v2's own routes still raise, so it fails loudly exactly where
  it matters.
- **Every string in the check is ASCII.** The first version used `→`, `—` and
  `§`; printing it on a Windows cp1252 console raised `UnicodeEncodeError` —
  a diagnostic that crashed while reporting the problem it existed to explain.
  The startup log line has the same constraint for the same reason.

The error text names *restarting* as the fix, not editing config. Last time the
config was already correct, and a reader who doesn't know that will change a
file that needs no changing and conclude the check is broken.

Guarded by `tests/t_live_backend.py` (14 checks), including that
`DeepInfraHybridBackend` actually overrides its parent's `"groq"` — without
that override the whole check passes vacuously.

---

## 6. Speech

### 6.1 The tick was split into bookkeeping and a speech decision

Reported three times: the model reads a label into the document and says
nothing.

A single call was being scored on two objectives that pull against each other —
keep the document accurate (which rewards quiet bookkeeping) and decide whether
to speak (which rewards noticing the user). **Silence kept winning, because it
was also the stated default for the other job.**

Now: Stage 1 does bookkeeping and its prose is discarded. Stage 2 gets a
separate, deliberately cheap prompt — no tools, no system brief, no full
document, no history, roughly a third of the size — and answers one question.

Phase 3c skips Stage 2 entirely when nothing could warrant speech, so an idle
tick still costs exactly one reasoning call.

### 6.2 Answering the user must not silence the follow-up

`MIN_UNPROMPTED_GAP_S` (90s) is the politeness budget. Speaking used to reset
it, which produced this:

> Asked to find the onions, the assistant replied *"I'll point them out as soon
> as they're in view"* — and **that reply gagged it for the entire 90-second
> search.** It found them at +27s, logged them silently, and said nothing until
> asked again.

A recent request is the one moment a follow-up is *solicited*. So a user turn
now opens `FOLLOWUP_WINDOW_S` (180s) instead of closing a gap.

`last_user_turn_ts` is tracked separately from `last_spoken_ts` on purpose: the
two pull in opposite directions. **Speaking should make the assistant quieter;
being asked should make it more forthcoming.**

`in_followup_window()` is deliberately **not** folded into
`may_speak_unprompted()`. `poll()`'s stale-task nags route through that
function, and a nag firing seconds after a conversation is exactly what the gap
is right to suppress. Only the tick path — where the model is reacting to
something it can actually see — opts in.

### 6.3 `[URGENT]` bypasses the gate and nothing else

One flag, one consequence. It does not change capture detail, does not affect
whether questions are asked, and does not resolve anything. Reserved for
physical risk and work about to be ruined — the cases where a 90-second wait
makes the warning worthless.

Confirmed live on 2026-08-10: *"⚠️ Slow down—path is tight by the door and
shelving"*, fired off a motion-blurred frame while the user was walking.

### 6.4 A dangling plan gets repaired deterministically

Observed live: the model replied *"Got it — we just need the tadka. Here's the
plan:"* and stopped, because the plan itself went into `update_tasks`. That
tool is `needs_followup=False`, so no second call happened and the naked
preamble shipped. **The user is listening, not reading** — a task list that
exists only in the doc panel does not reach someone whose hands are in the dal,
and they had to ask "I cannot see the plan".

`_repair_dangling_plan` fires on the exact shape of the failure — trailing
`:` / `—` plus a plan tool in the results — and appends the step count and the
first step. Deterministic rather than another model call: it costs nothing and
cannot itself dangle. **Only the first step is spoken**; reading six steps
aloud is how you lose someone at a stove.

For a proposal it appends *"Shall I go with that?"* instead — a dangling
proposal is the worse version, since the plan is not only unseen but waiting on
an answer the user was never asked for.

### 6.5 A user turn is never allowed to be silent

`require_text=True` is the difference between a tick and a user turn. On a tick,
empty text *is* the answer. On a user turn it never is.

Without it, a turn that did all its work through tools reported itself as a
failure. Observed live: asked for help with chole, the model wrote a correct
seven-step plan into the document, emitted no prose, and the user saw *"(no
reply — something went wrong, try again)"* beside a perfectly built plan.

There are four escalating rescues, in order:

1. **Empty response retry, with tools.** DeepSeek intermittently returns
   neither text nor tool calls. Retried *with* tools because the turn still
   needs to do its work — falling straight through to a text-only rescue loses
   the tool calls and the advice never reaches the document.
2. **The speechless follow-up.** Tool results are fed back with *"you said
   nothing to the user, who is waiting and cannot see tool calls — now SAY it"*.
3. **A forced text-only call, `tools=None`.** This is the whole point: while
   tools are offered the model can always answer with another call instead of
   prose. Taking them away leaves it nothing to reply with except words.
   Reproduced on "walk me through changing an oil filter", where a ReAct chain
   spent both earlier calls on tools.
4. **`_fallback_from_work`.** Reporting "something went wrong" beside a
   correctly built plan is worse than saying nothing useful — it tells the user
   to redo work that already succeeded.

The follow-up instruction also carries an anti-duplication clause: *"Anything
you already recorded is recorded… Re-recording produces duplicate watches, and
every duplicate is a question asked on every frame for the rest of the
session."*

### 6.6 The model is told how its own machine works

It knew it had "a persistent world document" and nothing else — not where the
observations come from, not that it cannot see, not that its own writes are
what it reads back next tick. Without that it cannot reason about its role; it
treats each tick as an isolated question rather than one step in a loop it is
steering.

`SYSTEM_BRIEF` is deliberately short — it is prepended to every tick, so each
sentence is billed a few hundred times an hour. Its last clause is the one that
matters most: *"Recording something is never the same as answering someone."*

---

## 7. The client (`live.js`)

### 7.1 The diff gate is the main cost control

32×32 grayscale, mean absolute delta. If the scene hasn't meaningfully changed,
no request leaves the browser. A tick firing mid-request is buffered as
`pendingFrame`, not dropped — latest matters.

Known limitation, inherited from v1: **the gate dies while walking.** Every
frame differs, so it skips nothing, and shopping burns budget far faster than
cooking.

### 7.2 A black frame is an unchanged frame

Reported live: camera view entirely black, system cheerfully reporting "nothing
changed". Correct, and useless — to a delta comparison a dead camera is
indistinguishable from a perfectly still scene.

Fixed with a **standard deviation** liveness test (`frameIsFlat`, threshold
2.0) that runs **before** the diff gate. A persistently flat frame gets a red
capture border, a distinct warning, and track diagnostics (`readyState`,
`enabled`, `muted`, dimensions).

Startup was hardened at the same time: `loadedmetadata` is bound **before**
`srcObject`, and `video.play()` is awaited explicitly with a tap-to-start
fallback, because an autoplay refusal leaves a black poster frame that looks
exactly like a dead camera.

**Caption reuse (§7.5) made this worse and that is why the two shipped
together** — before it, every chat turn sent a frame, so a dead camera still
got looked at.

Note what still isn't solved: flat frames are detected and reported, but they
are **still captioned at full cost**. The 2026-08-10 session spent five vision
calls describing a blackout.

### 7.3 One `busy` flag defeated every server-side fix

The lock restructure (§2), the yield points (§2.4), the phase split — none of
it reached the user, because `live.js` held a single `busy` flag covering both
ticks and chat:

```js
if (busy) {
  queuedPrompt = prompt;
  setStatus('queued — waiting for the current tick to finish…');
  return;
}
```

**The message never left the browser.** The status line was apologising for a
constraint that existed nowhere else in the system.

Split into `tickBusy` and `chatBusy`, protecting genuinely different things:
frames must not stack, and replies must not interleave. A *second* question
still queues behind the first — with voice input there is no visible input box
in which to notice a dropped message.

The lesson generalises: **a server-side concurrency fix is not done until the
client can exercise it.** Both halves need a test.

### 7.4 Renders must be applied in rev order

Once ticks and chat overlap, replies arrive out of order. A slow tick's render
predates a chat turn's writes and will stamp over it in the panel, silently
rolling the user's turn back on screen while the document on disk is fine.

`updateDoc(rendered, rev)` drops any render whose rev it has already passed.
See §2.5 for why the rev is bumped on window entry.

### 7.5 Caption reuse

An idle chat turn was paying for a fresh caption it didn't need. `live.js`
reuses the last one when the scene hasn't moved and it is under
`CAPTION_REUSE_MS` (15s).

The server tells the model this explicitly rather than leaving it to infer:
*"[No new camera frame this turn — the scene has not visibly changed since the
last observation above… That description still holds; answer from it.]"* The
alternative is a model that assumes it is blind when it is merely not looking
again.

Confirmed live 2026-08-10 (`capture: user turn — scene unchanged, reusing the
last caption`).

---

## 8. Rejected designs

### 8.1 A multi-agent orchestrator

Considered and rejected for v1, and the reasoning holds here. The reasoning
model orchestrates itself: it reads the scene, decides mid-thought whether to
call a tool, and routes to the right response type. The thinking chain *is* the
orchestration. See `docs/v1/DECISIONS.md` §9.

### 8.2 A VideoLLM-Online-style streaming architecture

Examined seriously (2026-08-10) as a route to real-time. Rejected — the
comparison clarified what v2 already gets right.

VideoLLM-Online encodes each frame through CLIP ViT-L/14 to 256 patches, then
**pools to a single vector**, and appends it to a growing context with a
streaming EOS token deciding whether to speak. Two structural problems for this
product:

- **One pooled vector is an unconditional compression.** It cannot preserve a
  label, a torque figure, or where fingertips sit relative to a blade. v2's
  caption is a *query-conditioned* compression — the brief tells the vision
  model what matters before it looks — which is why read mode can transcribe a
  packet at all.
- **Its KV cache grows without bound**, and decoding is memory-bandwidth bound,
  so latency degrades over a session. v2 never accumulates image tokens: each
  frame is captioned independently and only text persists, bounded by
  `RECENT_MAX` with compaction behind it.

The genuine lesson taken from it: **speech latency is not a real ceiling** —
buffered/streaming TTS lets the assistant talk while the next frame is being
processed. That is what `/v2/chat/stream` is for.

### 8.3 A hard cutoff on stale watches

Rejected in favour of the nudge-plus-degrade in §4.9. Silently dropping a watch
the user is still waiting on is worse than asking one stale question.

---

## 9. Still open

- **`/v2/chat/stream`** — SSE, client flushing to `speak()` on sentence
  boundaries. **Design constraint: one `_chat_turn` with an optional emit
  callback, not a second parallel function.** v1's
  `_process_locked`/`_process_stream_locked` divergence is the thing to avoid
  inheriting (`docs/v1/DECISIONS.md` §5.2).
- **Don't caption a flat frame** (§7.2) — detected but still billed.
- **The dangling-reply guard is too narrow** (§6.4) — it requires a tool from
  `_PLAN_TOOLS`, so a turn that trailed off after `log_environment` shipped
  with a bare colon on 2026-08-10.
- **`fine` detail has never been used live.** Every caption in the one real
  session is `coarse`, including reading a patent binder and searching a
  fridge. The tier passes its harness; the model never reaches for it.
- **Pipeline the two stages across frames** (§2.6) — vision captioning frame
  N+1 while reasoning digests frame N, taking a tick from `vision + reasoning`
  to `max(vision, reasoning)`. Designed, not built. The finding worth keeping is
  that it needs **no locking change**: the lock already handles overlapping
  turns, so the work is entirely client-side. Pairs with the backoff item below
  — one raises the ceiling, the other lowers the floor.
- **Adaptive tick backoff** — nothing throttles a 20-minute simmer, and the
  diff gate is useless while walking (§7.1).
- **Partial-evidence task completion** — a compound step ("Prep: dice onions,
  dice tomatoes, make paste, cut chicken") can be marked done from a frame
  showing one of four. Needs sub-items.
- **Per-user timezone.** `DEFAULT_TIMEZONE` is one server-wide setting standing
  in until there is somewhere to store preferences at all.

---

## Testing notes

`python tests/run_all.py` — 13 ad-hoc harnesses, no automated test suite.
v2-specific ones:

| | |
|---|---|
| `t_live_prompts.py` | prompt assembly, section by section |
| `t_live_parallel.py` | tick/chat overlap at both interruption points, **timed** |
| `t_live_writes.py` | deferred tool calls actually land on disk; proposal lifecycle |
| `t_live_backend.py` | the provider guard — §5.3 |
| `t_live_concurrent.js` | the client half of §7.3 and §7.4 |
| `t_live_blank.js` | `frameIsFlat` and the blank path |

The JS harnesses drive the **real** `live.js` in a stubbed DOM rather than
reimplementing it. Trap: top-level `let` in a `vm` script is not a property of
the sandbox object, so state must be read and poked via `runInContext` — a
harness that ignores this passes against a broken file.

**Harnesses cannot tell you whether the model behaves.** They cover what is
checkable without a live session; everything about *judgement* — does it
propose instead of writing, does it revert to coarse, does it speak when the
user is waiting — needs a real run and a read of the export.

**The export is the highest-value artifact.** It carries every vision prompt
and answer, silent ticks included, plus the world document as it stood at
export. Read it top to bottom: no single turn shows a flip-flop, only the
sequence does.
