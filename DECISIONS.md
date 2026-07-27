# Failure log & design decisions

Why the system is shaped the way it is. Every entry below is a real bug or a
real constraint hit in testing — not hypotheticals. **Read the relevant section
before changing that area**, or you will re-introduce something that was already
paid for once.

`CLAUDE.md` is the lean working reference. This file is the reasoning behind it.

---

## Contents

| Area | Jump |
|---|---|
| Provider limits & cost | [§1](#1-provider-limits--cost) |
| Model output handling | [§2](#2-model-output-handling) |
| Tool calling | [§3](#3-tool-calling) |
| State & concurrency | [§4](#4-state--concurrency) |
| The camera round trip | [§5](#5-the-camera-round-trip) |
| Live watching & silence | [§6](#6-live-watching--silence) |
| Voice-first behaviour | [§7](#7-voice-first-behaviour) |
| Frontend traps | [§8](#8-frontend-traps) |
| Rejected designs | [§9](#9-rejected-designs) |
| Still open | [§10](#10-still-open) |

---

## 1. Provider limits & cost

### 1.1 Groq's 8,000 TPM cap drove the whole backend split

Combined input **and** output, per minute. `max_tokens=8192` alone exceeded it
before counting the prompt. Everything shared one pool: image, history, tool
schemas, task list.

Fixed in stages:
1. `max_tokens` down to 4096 (thinking) / 1024 (not).
2. Live-tick requests shrunk (`871ffe8`).
3. **The real fix** — the hybrid split (`53f43b3`): Groq does *vision only*,
   DeepSeek does all reasoning and tool calling in text. Only the small
   image-only prompt has to fit under Groq's cap; the reasoning call carries
   whatever history it needs on a provider with no comparable ceiling.

### 1.2 The 200,000 TPD cap is what actually kills sessions

A 47-minute session died at ~198k/200k tokens **per day**, three consecutive
429s, unrecoverable until midnight. This is a different limit from the TPM one
and is not mentioned in most of the older notes.

In the hybrid, Groq does vision only — so **image tokens are essentially the
entire Groq budget**, and they scale with resolution, not with JPEG file size.

- Live ticks capped at `LIVE_FRAME_DIM = 640`; typed questions keep
  `MAX_FRAME_DIM = 1024` (detail matters when a human is waiting).
- `deepseek_backend.vision()` logs `Groq vision usage:` per frame. Before this,
  per-frame image cost was literally unmeasurable.
- **JPEG quality is not a lever.** Vision APIs bill by resolution/tiles.

**Known ceiling:** continuous all-day vision has a real paid floor. Resolution
is a constant-factor win, not an escape.

### 1.3 Render free tier sleeps after ~15 min of no *inbound* traffic

A server-side call *out* to Groq does nothing to prevent it, and can't run at all
once the process is dead. The frontend poll loop is what keeps it warm during
active use. A timer firing while the phone is closed is delayed until the next
request wakes the dyno.

---

## 2. Model output handling

### 2.1 Reasoning leaks into visible text

Qwen doesn't reliably close `<think>` tags, especially when cut off mid-thought
by `max_tokens`. An unclosed tag means tag-stripping finds no match and the raw
monologue becomes the "answer".

Fixed by requesting `reasoning_format: "parsed"` from Groq (separate
`message.reasoning` field) and trusting `VisionResponse.reasoning` directly.
Local Ollama-hosted Qwen3 still uses the inline-tag path.

### 2.2 Truncation has two distinct shapes — treat them differently

`VisionResponse.truncated` comes from `finish_reason == "length"`.

| Shape | Recovery |
|---|---|
| Reasoning **present**, text empty — ran out of budget after thinking, before answering | Feed its own reasoning back and ask it to conclude, `think=False`. Cheap; nothing re-derived. |
| Reasoning **absent/garbled** — nothing worth reusing | Fresh retry with `think=False`. |

Bounded to one retry either way.

**A correction worth keeping:** a populated reasoning field does *not* prove
`.text` is clean — a mid-stream cutoff can leave reasoning partial while `.text`
still holds leftover deliberation. The retry condition is `if response.truncated:`
unconditionally.

---

## 3. Tool calling

### 3.1 Native for Groq/DeepSeek, regex-parsed elsewhere

Originally *all* tool calls were regex-parsed from a ` ```tool {...}``` ` block
the model wrote into its own visible text. Free text has no schema — which is
exactly how §3.2 happened.

Both current backends set `SUPPORTS_NATIVE_TOOLS = True` and pass
`tools=`/`tool_choice="auto"`. Others default to `False`, accept a `tools` kwarg
for interface uniformity, and ignore it. `agent.py` branches on
`response.tool_calls` (native) vs `_execute_tool_calls(text)` (regex); both
converge on `{"tool", "arguments", "result"}`, so everything downstream is
agnostic.

`_build_reason_prompt` omits the "write a ```tool block" instructions when the
backend is native — telling a native tool-caller to *also* hand-write JSON just
invites a redundant malformed block.

### 3.2 Field-name drift silently emptied the task list

The model wrote `task` instead of `content`. `set_document()` dropped every item
with no error. Because `update_task_list` is a **full replace**, one malformed
call wiped everything.

Worse, the failure cascaded invisibly: `render_summary()` returns `""` for an
empty document, so `[Task list]` never got injected, so the live-frame silence
instruction never got injected — which is why the model "described the scene
again and again". **A silent state wipe looked like a completely unrelated
narration bug.**

Three layers of defence now:
1. Real nested JSON Schema on `items` (the structural fix).
2. `content`/`task`/`label` aliases in `set_document`.
3. Refuses to replace a populated document with an empty one.

### 3.3 Malformed tool calls used to crash the whole turn

A call missing its `"arguments"` wrapper either called the tool with zero args
(uncaught `TypeError`) or silently did nothing. Now: a missing `"arguments"`
falls back to the other top-level keys, and `TypeError` is surfaced as a normal
tool result string.

### 3.4 `needs_followup` is a cost control

Default `True`. Set `False` for tools whose result is a pure confirmation
(`start_timer`, `update_task_list`, `cancel_timer`, `log_observation`) — otherwise
you pay a second call just to restate your own confirmation. Tools that surface
genuinely new information (`web_search`, `fetch_page`) keep it `True`.

### 3.5 Only scan *visible* text for tool calls, never `<think>` blocks

The model mentions tool syntax hypothetically while reasoning about whether to
use one. Scanning the thinking trace turned that into a false invocation.

---

## 4. State & concurrency

### 4.1 Timers store wall-clock, not `asyncio.sleep`

`start_time + duration`, recomputed on each check. Render can restart the process
mid-wait; a sleeping task doesn't survive that, arithmetic does. Due-ness and
progress are **zero LLM cost**; the only call happens once, when one is found due.

### 4.2 Timer double-fire

The poll route and a live tick could both pick up the same due timer and fire two
completion messages. `mark_firing()` claims it **before** the awaited completion
call. Same lesson later applied to v2's trigger engine.

### 4.3 One lock, and `_locked` internals

Live ticks, typed chat, `request_camera` follow-ups and the timer poll all hit one
global agent, and every path `await`s real network calls — so two turns really
could interleave reads/writes of the task-list file.

`ChitraguptAgent._lock` (an `asyncio.Lock`, **not** reentrant) wraps the public
entry points. Internal call sites must use `_process_locked` /
`_check_timers_locked` or they deadlock on re-acquisition.

### 4.4 `found` was doing three jobs — the worst overload in the codebase

`log_observation(found=True)` simultaneously: (a) guaranteed a spoken alert,
(b) marked the task item `completed`, (c) closed the camera.

And it was *documented to the model* as "important enough to guarantee they're
told" — so it correctly set it on ordinary progress notes. One real session
marked **"Prepare tadka" complete** off a note reading *"oil poured into an empty
pan, no ingredients added yet"*, and shut the camera off. The user spent the rest
of the session asking why the camera kept closing.

Split into two flags:

| flag | speaks | completes item | closes camera |
|---|---|---|---|
| `alert=true` | ✅ | ❌ | ❌ |
| `found=true` on an ordinary step | ✅ | ❌ | ❌ |
| `found=true` on a `Find X` goal | ✅ | ✅ | ✅ |

Guarded in two places — item completion in `tasklist.add_observation`, camera
closing in `ChitraguptAgent._goal_complete` — both keyed on the item genuinely
being a find-goal. A stray `found=true` is logged, spoken, and **explained back
to the model** in the tool result so it corrects itself.

> **Rule:** one flag, one consequence. If a boolean drives both a user-visible
> action and a state mutation, split it.

### 4.5 Find-goals are sticky by design

`set_document` re-appends an omitted `in_progress` "Find X" item. Tool execution
order within a turn isn't guaranteed, so a goal registered by
`request_live_search` could otherwise be wiped by an `update_task_list` written
before it in the same completion. Consequence: a find-goal can outlive the turn
that created it — anything reading in-progress items must expect that.

---

## 5. The camera round trip

### 5.1 Typed questions never carried a frame

`sendMessage()` sends `currentImageBase64`, which is **only ever set by the file
picker** — never a camera frame. So the model was answering "where's the ice
cream" from stale text with no image at all.

Hence `request_camera`: a two-phase round trip, since the server can't reach into
the browser's camera mid-call.

- **Phase A** — model calls `request_camera`; `_process_locked` short-circuits and
  returns `{needs_camera: true}` *without* the usual follow-up logic.
- **Phase B** — client captures a frame and resends the same prompt with
  `is_camera_followup: true`.

`is_camera_followup` also suppresses the Phase B user-memory write, or the
identical resent prompt gets recorded twice.

### 5.2 The double-fire loop

Every camera turn fired `request_camera` **twice**. Two independent bugs:

**Server:** `_process_stream_locked` (the typed-chat path) passed
`self.tools.to_openai_tools()` — every tool, unconditionally — while
`_process_locked` filtered through `_available_tools`. So `request_camera` was
still on offer during Phase B, and with `has_image=false` the prompt still said
"call request_camera". The model obeyed.

**Client:** `startCameraStream()` returned the instant `srcObject` was assigned,
but `video.videoWidth` stays `0` until the first frame decodes. So
`captureCurrentFrame()` returned `null` — *that's why* Phase B kept arriving
frameless. The retry path passes the frame through with **no null check**.

Fixed both:
- Stream path filters identically, **and** camera tools are suppressed on any
  followup regardless of `has_image` — so even a frameless Phase B answers
  instead of looping.
- `startCameraStream` now `await video.play()` then awaits `waitForVideoReady()`
  (polls `videoWidth > 0`, 3s cap).

**Bounded, not runaway:** the retry callback discards its return value and never
re-checks `needs_camera`, so it stopped at exactly one wasted call each time. Both
wasted calls were `has_image=false` → **no Groq vision cost**, only DeepSeek
tokens. It did not cause the TPD crash; it cost ~9s of dead air and two
*"Let me take a look."* messages with no answer.

> **Rule:** the streaming and non-streaming paths duplicate a lot. Any change to
> tool availability, recovery, or response shape must land in **both**.

### 5.3 Still wasteful

`sendMessage()` should just attach the current frame when `cameraStreamActive` is
true, instead of a full extra round trip for pixels the browser already holds.
Not done.

---

## 6. Live watching & silence

### 6.1 The repetitive-narration failure

Classic VideoLLM-online symptom: *"everything remains exactly the same"*, every
tick. Two mechanisms fixed it.

**Silence protocol** — on `is_live_frame=True` with an active task item, if
nothing in the frame is new or relevant, the entire visible reply must be exactly
`SILENT_MARKER` (`"[SILENT]"`). `_process_locked` strips it to `""` before it
reaches the client.

Deliberately gated on `is_live_frame`: a direct user turn is **never** allowed to
go silent — someone is waiting on an answer.

**Observation log** — each task item carries `observations: list[str]` (capped at
`MAX_OBSERVATIONS_PER_ITEM = 5`). This is the substitute for VideoLLM-online's
KV-cached frame history: hosted APIs can't reuse image tokens across calls, so
each frame is converted to a short text fact once, the pixels are discarded, and
only the *text* accumulates. That's what lets "where is X" be answered from
history rather than only the current frame.

### 6.2 The silent pre-filter that ate frames

`_is_relevant_tick()` fired a cheap yes/no Groq call ("Is this frame relevant to
{goal}? yes/no") *before* the real reasoning call. If it said no, the tick
returned empty — no observation, no reasoning, nothing.

Indistinguishable on the client from a legitimate `[SILENT]`, so a frame actually
showing the target could be dropped by a crude misjudgment **with zero trace**. It
also doubled Groq calls per tick whenever a find-task was active.

**Removed entirely, with no replacement.** The perceptual diff gate (client-side)
and the model's own silence protocol are the cost controls. Relevance is decided
in one place — the documented pipeline — and nowhere else.

> **Rule:** never add a cheap pre-filter whose "no" is indistinguishable from a
> legitimate silence. Failures must be traceable.

### 6.3 The vision prompt was inferred, never requested

`agent.py` read `in_progress` items and picked one of three hardcoded prompts. The
reasoning model — the only thing that knows the goal — never said what it wanted.
That guessing is exactly why the `"Find "` prefix check exists: feeding a cooking
step to the detector produced *"find: Soak urad and rajma dal"*.

Task items now carry optional **`watch_for`**: the model's own brief to the vision
stage, preferred over every inference, with the old branches kept as fallback.

```
watch_for → find_targets → step description → generic gist
  (new)          (existing fallbacks, unchanged)
```

**The stage boundary matters.** Qwen sees pixels and knows nothing about the task,
the conversation, or what "correct" looks like. DeepSeek has all of that and no
pixels. So the brief must ask for *observations*, and the reasoning model supplies
the judgement:

```
watch_for: "Describe how the onion is being cut — slice thickness, how even,
            and roughly what fraction is still uncut. Say plainly when done."
Qwen     → "Sliced at roughly 5mm, uneven toward the root end. About half left."
DeepSeek → "Running a bit thick for a bhuna — aim closer to 3mm. Halfway there."
```

The wrapper enforces this: report facts and measurements, never advice.

Still **one** vision call per frame — the brief is standing state read back from
the task list, not a per-tick round trip asking what to look for.

An unbriefed find-goal is appended to a briefed request, or it would silently stop
being searched the moment another step got a brief.

### 6.4 Detail tiers — the decision has to happen before capture

640px is fine for "is the thing there / what is happening" and useless for
reading a label. But raising resolution alone would not have worked: the vision
wrapper explicitly said *"no full-scene description, four sentences"* and ran at
`max_tokens=160`, so a full-res frame still came back as a terse gist.

The binding constraint is that **resolution discarded in the browser can never be
recovered server-side**. So the level cannot be chosen when the model realises it
needs detail — by then the frame is already 640px. It has to be chosen *before
capture*, which means the decision must reach the client.

Task items carry `detail: "coarse" | "fine"`. When any in-progress item asks for
`fine`:

- the server echoes `frame_detail` on **every** response;
- the client sizes its **next** capture at `MAX_FRAME_DIM` instead of `LIVE_FRAME_DIM`;
- the vision wrapper gains a close-up instruction — read text, quote exactly,
  say which part was unreadable rather than guessing — and `max_tokens` goes
  160 → 500.

All three have to move together. Full-res pixels with a gist prompt and a 160-token
cap buys nothing.

**Cost is bounded by scope, not by a counter.** `fine` costs roughly 2.5x per
frame, and it is tied to one `in_progress` item — completing the step ends the
cost. A `pending` item asking for `fine` deliberately does *not* raise it, or
planning a close look later would make every frame expensive now.

**Not built:** a one-shot `inspect_detail(question)` tool for "look closely right
now" without a standing goal. It needs a `request_camera`-style round trip using a
full-res frame retained client-side, since the tick frame in hand is already
downscaled. The standing-brief route covers the continuous case (shopping,
inspection) and needs no new round trip, so it went first.

### 6.5 Flip-flops — independent calls with no continuity

Consecutive frames of a scene that had barely changed came back described
differently, and the reasoning model logged the contradictions as fact. Seconds
apart on the same pan, from one real session:

```
"Cumin seeds and chopped red onions are sauteing"
"Chopped onions and red chilies are being sauteed"      <- cumin gone
"The cumin seeds ... have not yet been added"           <- explicitly denied
"Cumin seeds or other spices are visible among the onions"  <- back
```

Two causes, so two rules, both appended after the branch that builds the prompt
so every variant gets them.

**Asserted absence.** The goal-aware prompt names the step's ingredients, which
turns the job into a roll-call — and a roll-call over small ambiguous items
(cumin among browned onions at 640px) produces confident false negatives. "Not
yet added" is a claim about a negative that one frame usually cannot support.
Absence is now only reportable when positively visible; otherwise say so.

**No continuity.** Each vision call is independent: no memory, no shared
attention. `FrameBuffer.last` already held the previous caption and the vision
stage simply never used it. It is now passed back as text, with an instruction
to describe change and not to contradict the previous frame about something no
longer clearly visible. This is the v1 counterpart of what `live/vision.py`
already does for v2 — the comparison baseline has to arrive in words, because
hosted APIs cannot share attention state across calls.

**Find-goals are exempt from both.** That branch has a strict one-line contract
where `NOT FOUND` is the correct answer for "I can't see it" — the entire point
of a search. The absence rule would fight it and the change rule would break
"nothing else". Guarded by `strict_detection`.

Grounding is also skipped for a one-off uploaded photo: the last caption there
is a stale live tick about something else.

### 6.6 Pipeline, end to end

1. `startCameraStream()` — `getUserMedia`, waits for a decoded frame.
2. `startLive()` — `setInterval(sampleLiveFrame, …)`, 2–15s (default 4s).
3. **Perceptual diff gate**, client-side, before any network call —
   `meanGrayscaleDelta()` vs the last *sent* frame. Skips the tick entirely.
   **This is the main cost control**, because it runs before the request leaves
   the browser.
4. Capture + send — `LIVE_FRAME_DIM` (640) for silent ticks, `MAX_FRAME_DIM`
   (1024) when a typed prompt rides along. JPEG q0.85.
5. **Busy-buffering** — a tick firing mid-request is held as `pendingLiveFrame`
   and flushed the instant the in-flight request finishes, not dropped.
6. Server: `LIVE_FRAME_MIN_INTERVAL_S` floor → `agent.process(...)`.
7. The lock (§4.3).
8. Prompt assembly — `[Camera feed]`, `[Timers]`, `[Task list]` + observations,
   then the silence instruction if a goal is active. Camera tools not offered.
9. Reasoning call; truncation recovery applies as normal.
10. Tool execution — usually `log_observation`.
11. Silence filtering — exact-match `[SILENT]` → `""`.
12. `_check_timers_locked()` under the same lock, so a timer firing mid-tick
    surfaces immediately.
13. Empty `final_text` → frontend renders nothing → loop.

`request_live_search` is not a different pipeline — it's a second *trigger* for
this one, registering `"Find X"` before the client does anything.

**Known gap:** nothing stops the loop once the target is found except a manual
camera-off or a genuine find-goal completion.

---

## 7. Voice-first behaviour

### 7.1 The "I updated the list" non-answer — self-inflicted

Asked *"what do I do next?"*, the model replied *"I've updated the list"* or
*"start with the in-progress item"*. Four times in one session.

The cause was **our own prompt**, which said: *"don't just recite the whole plan
back in your reply — the user can already see it."* That's false for a TTS voice
assistant. The user is listening, not reading; they can't see anything.

Three-layer fix:
1. That line deleted; replaced with "the list is read aloud, speak the concrete
   step".
2. Persona rewritten from a one-liner into a **role brief** — this came directly
   from the user noticing that *scolding the model once* made it behave, and
   asking for that to be permanent.
3. A deterministic backstop, `_is_list_meta_nonanswer` + `_voice_concrete_step`:
   detects a list-meta reply and makes one corrective call that forces the
   physical step.

`side_effect_silent` could **not** cover this — it only fires when visible text is
*empty*, and here the text was non-empty but useless.

### 7.2 The action-cue regex is a trap

`_is_list_meta_nonanswer` fires only when a reply is short, has list-meta
vocabulary, **and** has no action imperative (`_ACTION_CUE_RE`).

That regex held only kitchen verbs, so on a repair task a good answer — *"undo the
two screws and lift the panel off"* — read as actionless. Widened with hands-on
and screen-work verbs.

> **Trap:** `check`, `start`, `look`, `find`, `let`, `keep`, `move`, `read` are
> deliberately **excluded**. They occur inside the very non-answers this catches
> ("**check** the list", "**start** with the in-progress item"). Adding one
> silently disables the fix while every existing test still passes.

Widening is otherwise safe: more verbs only make the detector *less* likely to
fire, so a miss costs one needless corrective call.

### 7.3 Speak-after-task

A typed turn where the model called a side-effect tool but wrote no visible text
left a bare "⚡ Used tool" blob and nothing spoken — *"it made a task but never
said anything back."* `side_effect_silent` now does one follow-up call, gated to
non-live turns and only when the model actually went silent.

### 7.4 Timers were invisible to the model

It started them and never heard about them again — so it couldn't answer "how long
left on the eggs?" and had no way to name one for `cancel_timer`. A `[Timers]`
block now lists each label with time remaining. Pure arithmetic, ~zero cost.

### 7.5 `cancel_timer` deletes rather than marks fired

`mark_fired` is exactly what queues a completion message, so marking would
announce the timer you just cancelled. Already-fired-but-undelivered timers are
cancellable too, so a late "never mind" still suppresses it.

Matching is id → exact label → substring, and **ambiguity refuses**: cancelling
the wrong timer fails silently and the user only finds out when it never goes off.

### 7.7 Context loss — unresolved, now instrumented

Three consecutive turns with no memory at all. The user had just been given
detailed instructions; 36 seconds later:

> **user:** I have managed them now what should I do
> **assistant:** You'll need to give me a bit more context — what is it that
> you've managed? Are we working on a recipe, a project, a task list…?

Then *"it seems the task list didn't carry over to this conversation"*. It
recovered only once the user re-stated the dish, rebuilding the list from the
last few messages.

**Root cause is still unknown, and that is the finding.** Checked and ruled out:
live ticks polluting memory (they are excluded), the assistant reply not being
recorded (all paths record it), `/v1/reset` (it also clears `transcriptLog`, so
the export would not contain the earlier turns — it does), and Render
spin-down (the gap was 4 minutes, not 15). Both conversation memory *and* the
task list were lost together, which points at a process restart on an ephemeral
filesystem, but nothing confirms it.

It could not be root-caused after the fact because **nothing recorded how much
context a turn carried**. The only trace is the client wire log, bounded at
`DEBUG_LOG_MAX = 400`, which had already rolled past the window. "The model
ignored the task list" and "the task list was never in the prompt" are
indistinguishable from the transcript, and they need completely different fixes.

Both paths now log one line per turn:

```
turn context: history_turns=12 (sending 10) task_items=6
              task_list_in_prompt=True prompt_chars=3184
```

Cheap, and it makes the next occurrence self-diagnosing rather than
unfalsifiable. Deliberately not a speculative fix — guessing at a cause here
would most likely have added machinery that addressed nothing.

### 7.6 Don't start a timer on a planning statement

*"I want to boil eggs"* is not *"the eggs are on"*. Start one only once the step
has genuinely begun — user-confirmed or camera-visible.

---

## 8. Frontend traps

### 8.1 The service worker will hide your work

`sw.js` is cache-first for the app shell, keyed by `CACHE_NAME`. It only
re-fetches when **its own bytes change**. `CACHE_NAME` sat at `v1` for an entire
development session, so every browser that had ever loaded the app kept serving
the stale shell through every deploy — voice input, camera toggle and more were
silently invisible **despite being correctly shipped**.

> **Rule:** any change to `index.html` / `app.js` / `style.css` **must** bump
> `CACHE_NAME`. Currently `v16`. Server-only changes don't need it.

`/v2` and `/live` are excluded from the SW entirely so iterating there never
fights the cache.

### 8.2 `debug.js` duplicates `app.js`

Camera lifecycle, frame sizing and capture logic exist in both. Changes to one
almost always belong in the other.

### 8.3 Windows `uvicorn --reload` serves stale bytecode

Observed repeatedly: WatchFiles logs a reload, old code keeps serving. **Restart
manually** when iterating on `server/agent/*.py`. If behaviour doesn't match a
just-made edit, suspect this before suspecting the edit.

---

## 9. Rejected designs

**Multi-agent / orchestrator for task tracking.** One model + shared document
state instead of N workers + a coordinator. Splitting would mean paying a
coordination call on every check-in for no benefit over one model reading a shared
document.

**Pure orchestration (a controller routing between specialists).** Requires
predicting every situation in advance and breaks on edge cases. ReAct lets the
model orchestrate itself — the thinking chain *is* the orchestration.

**A per-tick "what should I look for?" call.** Would double calls on every frame
for something the model already told us via the task list. `watch_for` is standing
state, read back for free.

**Paid TTS (Groq/ElevenLabs).** Web Speech API is free, on-device, ~zero latency,
no server call and no token cost. A paid voice can be swapped in behind `speak()`.

**A replacement live-tick pre-filter.** See §6.2 — deliberately none.

---

## 10. Still open

Ordered by how much they hurt in real use.

| # | Item | Notes |
|---|---|---|
| 1 | **Adaptive poll backoff** | Fixed 4s tick regardless of context. Nothing throttles a 20-minute simmer. Bound below Render's idle timeout. |
| 2 | **The diff gate dies while walking** | Every frame differs, so it skips nothing. Shopping would burn TPD far faster than cooking. The cost model assumes a static scene. |
| 3 | **Context loss** | ~3-turn amnesia observed. Root cause still unknown — see §7.7. Diagnostics added; needs a recurrence to pin down. |
| 4 | **Attach the live frame in `sendMessage`** | §5.3. Removes the whole `request_camera` dance from the common case. |
| 5 | **One-shot `inspect_detail`** | Close look with no standing goal. Needs a client-retained full-res frame + a round trip. §6.4. |
| 6 | **Compound-item completion** | "Gather Ingredients" can be marked done from partial visual evidence. Needs sub-items. |
| 7 | **No pruning of completed items' observations** | Grows unboundedly across a long session. |
| 8 | **Render keep-alive pinger** | Only matters for unattended timers >15 min. |
| 9 | **Multi-timer UI progress** | `active` is already returned by `/v1/timers/check`, just unrendered. |
| 10 | **Egocentric fine-tuning** | Long-term. Identify real failure cases on real footage *first*, then fine-tune on those patterns only. Ego4D / Egocentric-1M / EPIC-Kitchens. |

---

## Testing notes

There is no test suite in the repo. Verification has been ad-hoc harnesses run
against temp state files — the most recent covers 48 checks across the flag split,
`watch_for` storage and precedence, `cancel_timer` matching and ambiguity, and a
regression set pinning the five real transcript non-answers.

**What actually finds bugs here is a real session, exported.** The debug UI's
`## Pipeline / wire log` section is the highest-value artifact: it shows every
POST with `is_live_frame` / `has_image` / `is_camera_followup`, the
`[vision→Qwen] asked:` prompt and its answer, every tool call, and the final
event. Most entries in this document were found by reading one.
