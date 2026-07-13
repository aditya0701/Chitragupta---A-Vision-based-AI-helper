# Vision Chitragupta

An egocentric live camera assistant with a two-stage vision + reasoning pipeline,
inspired by the Hindu mythological figure Chitragupta — the all-seeing record keeper
who observes, records, and reports.

The system watches through a camera, understands what it sees using a vision model,
reasons about it using a ReAct agent with chain-of-thought thinking, and responds
via text and optional TTS.

**Primary use case:** hands-free kitchen assistance over voice/text — e.g. it tells
you to boil eggs, starts a timer itself, keeps track of everything else you're doing
in parallel (chicken prep, veggies), and tells you when to come back to something —
without you having to ask it or re-explain state it should already remember.

---

## Current implementation status

This is a working FastAPI server (`server/`), not just a design doc. Core pipeline,
tool calling, background timers, and task-list tracking are built and tested against
the live Groq API. Voice input (speech-to-text) was added 2026-07-13 via the
browser's Web Speech API. Not yet built: TTS output, multi-timer UI progress
display, adaptive poll backoff. See TODO at the bottom for the real current list.

---

## Architecture

```
Phone/browser  →  /v1/chat (FastAPI)
                     ↓
              [Backend]                Pluggable: colab | api | local
              Vision + reasoning       (see server/backends/)
                     ↓
              [Tool execution]         start_timer, update_task_list,
              Parses ```tool {...}```  web_search, fetch_page, calculate, get_time,
              blocks from the reply   log_observation, request_camera (added 2026-07-13)
                     ↓
              [Response]               Text (+ think_blocks for debug UI)
```

Two backend shapes exist, selected by `SPLIT_VISION_REASONING` on the backend class:

- **Split (Colab)**: a dedicated vision model (Qwen3-VL) describes the frame first,
  then a separate reasoning model (Qwen3) thinks and responds. Two model calls.
- **Single-call (API mode — Groq/Gemini/OpenAI/Anthropic)**: one multimodal model
  sees the image and reasons in the same call. This is the active configuration —
  no reason to pay for two calls when one multimodal model does both.

### Why this split (when using Colab)
The vision model's only job is turning pixels into words. All intelligence lives in
the reasoning layer, which only runs when something is worth responding to.

### Why ReAct over pure orchestration
Pure orchestration (a controller that routes between specialist models) requires
predicting every situation in advance and breaks on edge cases. ReAct lets the
reasoning model orchestrate itself: it reads the scene, decides mid-thought whether
to call a tool, and routes to the right response type without a separate controller.
The thinking chain IS the orchestration. This same reasoning is why multi-task
tracking (see below) uses one model with shared state rather than an
orchestrator-plus-worker-agents pattern — splitting into multiple LLM instances
would mean paying for a coordination call on every check-in, for no benefit over one
model reading a shared document.

---

## Current stack

| Layer | Tool | Notes |
|---|---|---|
| Server | FastAPI (`server/main.py`) | Single global `ChitraguptAgent` instance |
| Active backend | Groq API (`qwen/qwen3.6-27b`) | `BACKEND_MODE=api`, `API_PROVIDER=groq` |
| Alt. backends | Gemini, OpenAI, Anthropic, Colab+Ollama, local Ollama | See `server/backends/` |
| Deployment | Render (free tier) | `render.yaml` — spins down after ~15 min idle |
| Frontend | Static HTML/JS/CSS (`server/static/`) | No build step, served directly by FastAPI |
| TTS | Not yet implemented | Planned: browser Web Speech API (free, zero setup) |
| Camera | Phone/laptop browser | MediaStream API, works over any HTTPS connection |

**Local dev:** `uvicorn server.main:app --host 0.0.0.0 --port 8000` from the repo
root (or via `server/venv`). Requires `TOOLS_ENABLED=true` in `server/.env` — it
defaults to `false`. Prefer running **without** `--reload` when iterating on
`server/agent/*.py` — WatchFiles has been observed serving stale bytecode on
Windows after edits; restart manually instead.

**Connectivity:** the phone does not need to share a network with anything. Once
deployed on Render, it's a public HTTPS URL — any internet connection (WiFi or
mobile data) works, same as opening any website. `localhost` only works for testing
on the same machine that's running the server (camera access requires either
HTTPS or the literal hostname `localhost` — a bare LAN IP over HTTP will fail the
browser's secure-context check for `getUserMedia`).

---

## Agentic tools & persistent state

Two pieces of state live outside the conversation history, both under
`server/data/` (gitignored, survives process restarts by design):

### Timers (`server/agent/timers.py`)
Started via the `start_timer(label, duration_seconds, context)` tool. Stores
**wall-clock `start_time` + `duration`**, not a running `asyncio.sleep` — this
matters because Render's free tier can restart the process mid-wait; recomputing
`elapsed = now - start_time` on every check is resilient to that in a way an
in-memory sleeping task is not.

- Checking due-ness and progress (`% done`) is pure arithmetic — **zero LLM cost**.
- The only Groq call happens once, when a timer is actually found to be due, to
  generate a contextual next-step message (not just "timer done" — it gets the
  original `context` string and current task list, so it can say something useful).
- That completion call is routed through the *same* prompt-building and
  tool-execution path as a normal turn (`ChitraguptAgent.check_timers()` reuses
  `_build_reason_prompt` / `_execute_tool_calls`), so completing a timer can also
  update the task list (mark a step done), not just narrate that it happened.
- `check_timers()` runs from two places: the background poll (`GET
  /v1/timers/check`, called by the frontend every 15s) for quiet stretches, *and*
  from the end of every `process()` call (i.e. every real `/v1/chat` turn) so a
  completion surfaces immediately if you're actively mid-conversation about
  something else when a timer fires, rather than waiting for the next poll tick.

### Task list (`server/agent/tasklist.py`)
A structured living document — title + items, each with `status`
(`pending`/`in_progress`/`completed`/`skipped`) and an optional `note` (used for
substitutions, e.g. "used tofu instead of paneer"). Modeled directly on Claude
Code's own `TodoWrite` tool: the model resends the **full item list** on every
edit rather than diffing — no separate add/remove/branch tools, the server just
persists whatever it's given. Completed items stay in the list (marked, not
deleted) so it doubles as a record of what happened, not just a queue of what's
left. Item `id`s are stable across edits (matched by content) so the same logical
item keeps its identity as its status changes turn to turn.

The current document is injected into **every** reasoning prompt as `[Task list]`
context (see `_build_reason_prompt`), so the model doesn't need to be reminded of
it and can act on it proactively — e.g. deciding *unprompted* mid-conversation
that it's time to start boiling eggs, not just reacting when told to.

### Observation log & goal-directed vision (added 2026-07-13)
Motivated by real Live-mode transcripts showing the VideoLLM-online-style
repetitive-narration failure ("everything remains exactly the same" every
tick), and a separate discovered bug where typed chat questions never had a
current camera frame attached at all — the model was answering "where's the
ice cream" from stale text with no image.

- Each task-list item can now carry `observations: list[str]`
  (`tasklist.add_observation(item_ref, note)` in `tasklist.py`, capped at
  `MAX_OBSERVATIONS_PER_ITEM = 5`, oldest dropped first). Rendered nested
  under the item in `render_summary()`. This is the substitute for
  VideoLLM-online's KV-cached frame history: we can't reuse image tokens
  across hosted-API calls the way a self-hosted model reuses its attention
  cache, so instead each frame is converted to a short text fact once, the
  pixels are discarded, and only the *text* accumulates across turns.
- `log_observation` tool (`needs_followup=False`) — the model calls this on
  every live-frame tick relevant to an in-progress task-list item, whether
  or not it also decides to say something out loud. This is what lets a
  later "where is X" question be answered from logged history instead of
  only the current frame.
- **Live-frame silence protocol**: `_build_reason_prompt` tells a
  `is_live_frame=True` turn that if nothing in the current frame is new or
  relevant to an active task-list item, its entire visible reply must be
  exactly `SILENT_MARKER` (`"[SILENT]"`, defined at the top of `agent.py`).
  `_process_locked` strips this to an empty string before it ever reaches
  the client. This rule is deliberately gated on `is_live_frame` — a direct
  user turn (typed message, or a `request_camera`-fulfilled turn) is never
  allowed to go silent, since the user is actively waiting on an answer.
  Old behavior (always narrate the full scene every tick) is untouched for
  turns with no active task-list item — silence only kicks in once there's
  something to be silent *relative to*.
- **`request_camera` tool** — for a direct text turn with no image attached
  (the `sendMessage()` bug above). The server can't reach into the client's
  camera mid-call, so this is a two-phase round trip instead of a normal
  tool: the model calls `request_camera`, `_process_locked` short-circuits
  and returns `{needs_camera: true}` *without* running the usual
  follow-up-call logic, the client (`captureCurrentFrame()` /
  `postChat()` in `app.js`) grabs the current live-video frame and resends
  the same prompt with it attached. If the camera isn't actually running
  (Chat & Image tab, no stream), the client tells the user to open Live
  Watch instead of retrying blindly. The tool is only offered in the prompt
  when `has_image=False and not is_live_frame` — live ticks and
  image-attached turns already have an image, so offering it there would
  just invite a redundant ask.
- **Frontend frame buffering**: `sampleLiveFrame()` used to silently drop a
  tick if a previous request was still in flight (e.g. a slow `web_search`
  tool call). It now stores the latest frame in `pendingLiveFrame` and
  `sendLiveFrame`'s `finally` block flushes it immediately once the agent
  is free, instead of waiting for the next interval tick — so a moment
  that matters isn't lost to a slow turn elsewhere.
- **Concurrency lock**: none of the above is safe without serialization —
  live ticks, typed chat, and the timer poll all hit the same global
  `ChitraguptAgent` with no prior synchronization, and since every path
  `await`s real network calls, two turns really could interleave their
  reads/writes of the task-list file. `ChitraguptAgent._lock` (an
  `asyncio.Lock`, not reentrant) now wraps `process()` and `check_timers()`
  entry points; internal call sites use the `_locked` variants
  (`_process_locked`, `_check_timers_locked`) to avoid deadlocking on
  re-acquisition. `timers.py` also gained `mark_firing()`, claimed *before*
  the awaited completion call, so the poll route and a live tick can't both
  pick up and fire the same due timer.

### Reliability fixes from real-transcript testing (added 2026-07-13, later same day)
A live test against the Groq API surfaced two failure modes the unit-level
checks didn't catch:

- **`content`/`task` field-name drift silently emptied the task list.**
  Tool calls are parsed from free-form JSON in the response text (see
  "Tool-calling is prompt-based, not native function-calling" below) — there
  is no schema enforcing the model always writes `content` for an item's
  text. When it wrote `task` instead, `tasklist.set_document()` dropped
  every item with no error, and because `update_task_list` is a full
  replace, this could wipe an already-populated document down to empty.
  Since `render_summary()` returns `""` for an empty document, this also
  meant `[Task list]` and the live-frame silence instruction never got
  injected into the prompt at all — explaining the "described the scene
  again and again" symptom directly. Fixed in `tasklist.set_document()`:
  accepts `task`/`label` as aliases for `content`, and refuses to replace a
  populated document with an empty one (logs a warning, keeps the old
  document) if every incoming item fails to parse.
- **Truncated reasoning leaked raw chain-of-thought as the visible answer.**
  A complex, imageless, multi-tool planning turn (list ingredients + build a
  task list + decide whether to request the camera) ran long enough under
  `reasoning_effort: "default"` to hit the `max_tokens=4096` ceiling before
  finishing its thought. Groq's `reasoning_format: "parsed"` only populates
  the separate `message.reasoning` field when the model *completes*
  reasoning — cut off mid-thought, the whole raw monologue lands in
  `message.content` instead, with no `<think>` tags for the existing
  stripping logic to find. `VisionResponse` gained a `truncated` field
  (`groq_backend.py`, from `finish_reason == "length"`); `agent.py` now
  retries once with `think=False` when a response is truncated with no
  separated reasoning, instead of showing the raw dump to the user.
- **Related latency/robustness fix, same root cause:** any imageless turn
  where `request_camera` is offered (see `offer_camera` in
  `_process_locked`) now forces `think=False` unconditionally, not just on
  truncation-retry. Reasoning hard before the model has even seen an image
  is mostly wasted effort anyway (it can't ground specifics without
  vision), and this was exactly the shape of turn most likely to run into
  the truncation bug above — bounding it there fixes both the "took too
  long" and "spewed its thinking" complaints from the same change.
- **Malformed tool calls no longer crash the turn.** `_execute_tool_calls`
  previously only caught `json.JSONDecodeError`; a tool call missing its
  `"arguments"` wrapper (e.g. `{"name": "update_task_list", "items": [...],
  "title": "..."}` instead of nesting under `"arguments"`) either silently
  called the tool with zero args (`TypeError`, uncaught, failed the whole
  turn) or, for optional-arg tools, quietly did nothing. Now: a missing
  `"arguments"` key falls back to every other top-level key in the call
  object, and a `TypeError` from `tool.fn(**arguments)` (wrong/missing
  required args) is caught and surfaced as a normal tool result string
  instead of crashing.
- **Groq token usage is now logged** (`groq_backend.py`) —
  `resp.usage.prompt_tokens`/`completion_tokens`/`total_tokens` plus
  `finish_reason`, logged on every call. Groq doesn't publish a per-image
  token cost for `qwen/qwen3.6-27b` in their docs; this is how to get a
  real number for this app's actual resolution/quality settings instead of
  guessing.

### Camera stream decoupled from Live Watch polling (added 2026-07-13, fifth pass)
Two user-reported gaps: `request_camera` failing silently with no camera
enabled just showed inert text ("open Live Watch first") with nothing
clickable, and there was no way to make the camera available for on-demand
questions without committing to Live Watch's continuous 4s polling loop —
the two were bundled into one `startLive()` call with no way to get one
without the other.

- **`app.js`**: split the raw stream lifecycle out of Live Watch's
  autonomous polling. `startCameraStream()`/`stopCameraStream()` now just
  handle `getUserMedia` + wiring `<video>` — no interval, no per-frame
  Groq calls. `startLive()` calls `startCameraStream()` first, then layers
  polling on top (`liveActive = true`, `restartLiveTimer()`). New state
  var `cameraStreamActive` (stream attached at all) is distinct from
  `liveActive` (autonomous polling specifically) — `captureCurrentFrame()`
  now checks the former, so it works whether the stream came from Live
  Watch or the manual toggle.
- **New manual toggle**: 🎥 button in the Chat & Image input row
  (`#camera-toggle-btn`, `toggleCameraStream()`) — turns the camera on for
  on-demand `request_camera` use only, never starts polling. Hidden while
  in Live Watch mode (that tab owns the camera lifecycle directly, showing
  both would be confusing about which one's in control). Never auto-starts
  — camera only turns on from an explicit click, in either mode.
- **`request_camera`-with-no-stream fallback is now actionable**:
  `addCameraEnableMessage()` renders a real `<button>` in the chat message
  ("🎥 Enable camera") that starts the stream and automatically retries the
  same question on click, instead of a dead-end sentence the user had no
  way to act on. `sendMessage()`'s response-rendering logic was extracted
  into `renderChatResponse()` so both the normal path and this retry path
  share it instead of duplicating the scene_unchanged/think_blocks/tool_calls
  handling.

### Second-opinion review fixes (added 2026-07-13, fourth pass)
An external review of the request_camera flow caught two real bugs the
earlier passes missed (and correctly cleared two others that looked
plausible but didn't hold up against the actual code):
- **Fixed — think=False was firing on substantive questions, not just
  trivial camera-routing decisions.** The old `offer_camera` check forced
  `think=False` for *any* imageless tool-enabled turn, which included "help
  me plan chicken biryani" — the one call where getting the recipe/step
  breakdown right matters most. Removed the override entirely;
  `think = should_think(prompt)` alone decides now, and the
  retry-once-on-truncation logic (previous pass) is the actual safety net
  for a turn that runs long, applied only when it happens rather than
  preemptively on every imageless question.
- **Fixed — the request_camera round trip double-recorded the user's
  message.** Phase A recorded the user's prompt *and* a placeholder
  assistant reply ("Let me take a look."); the client's Phase B resend of
  the identical prompt text triggered a second `memory.add("user", ...)`.
  Added `is_camera_followup` (threaded from `ChatRequest` through
  `process()`/`_process_locked`) — skips the user-memory write on Phase B,
  and the placeholder assistant write on Phase A was removed entirely
  (Phase B's real answer is now the only thing recorded for the exchange).
- **Checked and NOT bugs, despite looking plausible:** tool-call ordering
  between `update_task_list` and `request_camera` doesn't matter — both
  `_run_structured_tool_calls` and `_execute_tool_calls` execute every tool
  call in a response fully before the `request_camera` short-circuit check
  ever runs, so `update_task_list` always executes regardless of position.
  And the `[SILENT]` exact-match check doesn't conflict with `log_observation`
  firing on the same tick — tool-block stripping happens before the
  marker comparison for the regex path, and native tool calls never appear
  in `message.content` at all for the Groq path, so there's nothing to
  conflict with in either case.
- **Deferred, not fixed:** marking a compound task-list item (e.g. "Gather
  Ingredients") complete from partial visual evidence is still possible —
  would need per-ingredient sub-items to actually fix, separate scope from
  this pass. Live Watch still polls every 4s with no way to say "nothing to
  watch for the next 30 minutes" during a timer wait — no `start_watch`-style
  feature exists yet despite the idea coming up in review; would need to be
  designed, not just wired in.

### Tool-calling: native for Groq, prompt-parsed elsewhere (updated 2026-07-13)
Originally all tool calls were parsed via regex from a ` ```tool {...}``` `
block the model writes into its own visible response text
(`_execute_tool_calls`), ReAct-style — `ToolRegistry.to_openai_tool()`
existed but was dead code, nothing passed `tools=` to the API. This is what
let field-name drift happen at all (see above): free text has no schema to
violate.

Confirmed via Groq's docs that `qwen/qwen3.6-27b` supports native
function-calling (`tools`/`tool_choice` params, same shape as OpenAI's), so
`GroqBackend` now uses it: `SUPPORTS_NATIVE_TOOLS = True`, `chat()` passes
`tools=` + `tool_choice="auto"`, and parses structured `message.tool_calls`
into `VisionResponse.tool_calls` instead of relying on the model to
hand-write correctly-shaped JSON into text. `Tool.to_openai_tool()` was
fixed to emit valid JSON Schema (no more `required` leaking into individual
property definitions) and `update_task_list`'s `items` param now has a real
nested object schema (`content`/`status`/`note`) instead of only a prose
description — this is the actual structural fix for the field-name-drift
bug, with the alias/wipe-guard patches in `tasklist.py` as a second line of
defense.

Every other backend (Gemini, OpenAI, Anthropic, Colab/Ollama) still uses the
old regex-parsed path — `SUPPORTS_NATIVE_TOOLS` defaults to `False` for
them, they accept a `tools` kwarg (for interface uniformity, since agent.py
always passes `tools=self.tools.to_openai_tools() if TOOLS_ENABLED and
backend.SUPPORTS_NATIVE_TOOLS else None`) but ignore it. `agent.py` branches
on `response.tool_calls` (native, if present) vs. `_execute_tool_calls(text)`
(regex fallback) per turn — both paths converge on the same
`{"tool", "arguments", "result"}` shape, so everything downstream
(`request_camera` detection, `needs_followup` handling, filtering) is
unaffected by which path produced it. `_build_reason_prompt` skips the
"write a ```tool block" formatting instructions when
`self.backend.SUPPORTS_NATIVE_TOOLS` is true, keeping only the behavioral
guidance (when to use each tool), since telling a native-tool-calling model
to *also* hand-write JSON into text just invites a redundant/malformed
extra block.

### Cost-control patterns worth preserving
- `Tool.needs_followup` (default `True`) — set `False` for tools whose result is a
  pure confirmation (`start_timer`, `update_task_list`). Tools that surface new
  information the model hasn't seen (`web_search`, `fetch_page`) keep it `True`.
  This avoids a wasted second Groq call just to have the model restate its own
  tool-call confirmation in prose.
- Tool calls are only ever scanned from the **visible** response text, never from
  `<think>` blocks — the model sometimes mentions tool syntax hypothetically while
  reasoning about whether to use one, and scanning the thinking trace turned that
  into a false invocation.
- Multi-agent/orchestrator patterns were deliberately rejected for multi-task
  tracking (see "Why ReAct" above) — one model + shared document state instead of
  N worker agents + a coordinator, for the same reason the original controller-LLM
  design was dropped.

---

## Known constraints (learned the hard way, don't relitigate)

- **Groq account TPM cap:** this account's Groq tier caps requests at **8000
  tokens/minute combined input+output**. `max_tokens=8192` alone exceeded it before
  even counting the prompt. Currently `4096` when thinking is on, `1024` when off
  (`server/backends/groq_backend.py`) — enough headroom for this model's verbose
  reasoning without tripping the cap on typical prompt sizes. If prompts grow a lot
  (long conversation history, big task list), this may need revisiting.
- **Groq reasoning leaks into content without `reasoning_format="parsed"`:** this
  model doesn't reliably close `<think>` tags inline, especially if cut off by
  `max_tokens` mid-thought — an unclosed tag means the regex-based stripping finds
  no match and the raw reasoning trace becomes the visible "answer." Fixed by
  requesting `reasoning_format: "parsed"` from Groq (returns reasoning in a
  separate `message.reasoning` field) and trusting `VisionResponse.reasoning`
  directly in `agent.py` when a backend provides it, bypassing tag-stripping
  entirely for Groq. Local Ollama-hosted Qwen3 still uses the inline-tag path.
- **`TOOLS_ENABLED` defaults to `false`** (`server/config.py`) — disabled a few
  commits back while testing plain API chat. Nothing in the Tools section above
  works until `TOOLS_ENABLED=true` is set.
- **Render free tier spins down after ~15 min with no inbound HTTP traffic.**
  This is about traffic *to* the Render server specifically — a server-side call
  *out* to Groq does nothing to prevent this, and can't run at all once the process
  is already killed. The frontend's existing poll loop (camera sampling in Live
  mode, or the 15s timer-check poll) is what keeps it alive during active use; if
  the phone is closed for 15+ min mid-timer, the completion message is delayed
  until the next request wakes the dyno. Fix if needed: a free external pinger
  (UptimeRobot, cron-job.org, or a GitHub Actions scheduled workflow) hitting
  `/health` every ~10 min. Not yet set up.
- **Windows `uvicorn --reload` has served stale code after edits** during this
  project's development (WatchFiles detected the change and logged a reload, but
  the old bytecode kept serving). If behavior doesn't match a just-made edit,
  restart the server manually before assuming the code is wrong.

---

## UI modes (`server/static/`)

Two tabs, switched via `switchMode()` in `app.js`:
- **💬 Chat & Image** — manual text/image testing, tools visible in sidebar. Default.
- **📹 Live Watch** — auto-starts the camera, samples on an interval with a
  perceptual diff gate (skips the network call entirely if the scene hasn't
  meaningfully changed — this is what actually protects API quota, since it runs
  before any request leaves the browser), responds to changes automatically.

Switching tabs drives the camera lifecycle directly (starts/stops `getUserMedia`);
there's no separate manual toggle button anymore. Timer completions render as
normal chat messages (⏰ prefix) whether they arrive via the background poll or
folded into an active chat reply.

---

## File structure (actual, as built)

```
/
├── CLAUDE.md
├── render.yaml                  Render deployment config (free tier)
└── server/
    ├── main.py                  FastAPI app, routes (/v1/chat, /v1/timers/check, /v1/reset)
    ├── config.py                Settings from .env (BACKEND_MODE, TOOLS_ENABLED, etc.)
    ├── agent/
    │   ├── agent.py             ChitraguptAgent — main loop, prompt building, tool execution
    │   ├── __init__.py          Tool/ToolRegistry, built-in tools, ConversationMemory
    │   ├── timers.py            Persisted background timers (wall-clock, survives restarts;
    │   │                        mark_firing() claim added 2026-07-13 to prevent double-fire)
    │   └── tasklist.py          Persisted task/recipe document (Claude Code TodoWrite-style;
    │                            per-item observations + add_observation() added 2026-07-13)
    ├── backends/
    │   ├── __init__.py          VisionBackend ABC, VisionResponse, should_think() heuristic
    │   ├── groq_backend.py      Active backend
    │   ├── gemini_backend.py, openai_backend.py, anthropic_backend.py
    │   ├── colab.py             Split vision/reasoning via Ollama on Colab
    │   └── factory.py           get_backend() — picks backend from BACKEND_MODE/API_PROVIDER
    ├── static/
    │   ├── index.html, app.js, style.css   No build step
    │   └── manifest.json, sw.js            PWA support
    └── data/                    Gitignored — timers.json, document.json (runtime state)
```

---

## Prompts

The reasoning system prompt is built dynamically in `agent.py::_build_reason_prompt`,
not stored as a static file. It assembles, per turn: persona line, `[Camera feed]`
or attached-image note, `[Task list]` (if a document is active, now with nested
observations per item), the user's message, a thinking-mode instruction, and —
if `TOOLS_ENABLED` — the tool list plus tool-specific usage guidance (start_timer
is fire-and-forget; update_task_list is full-list-replace, keep completed items,
don't recite the plan back verbatim).

**Added 2026-07-13:** the prompt now branches on `is_live_frame` and `has_image`
(see "Observation log & goal-directed vision" above) — `request_camera` is only
listed as a tool when the current turn has no image, and the
silence/`log_observation` instruction is only injected on live-frame ticks with
an active task list. Direct user turns keep the original always-answer behavior
unchanged.

---

## TODO

Done this session: ReAct tool parsing, tool executor, timers, task-list tracking,
Groq reasoning-leak fix, two-mode UI.

Done 2026-07-13 (basic version, meant to be iterated on further): concurrency
lock across process()/check_timers(), timer double-fire fix, per-item
observation log + `log_observation` tool, live-frame silence protocol
(`[SILENT]` marker, scoped to `is_live_frame` only), `request_camera` tool +
two-phase client round trip for imageless text turns, frontend live-frame
buffering (stopped dropping frames while busy). Known follow-ups not yet
done: no pruning of a *completed* item's observations, no retry/backoff on a
repeated camera-unavailable response, `request_camera` and the silence
protocol have only been exercised with unit-level checks (fake backend), not
yet against the live Groq API end-to-end.

Done 2026-07-13 (third pass, after real Groq testing surfaced actual bugs):
see "Reliability fixes from real-transcript testing" above — task-list
field-name alias + wipe guard, truncated-reasoning retry, bounded reasoning
on imageless turns, hardened tool-call parsing, Groq usage logging. Also
added: browser-native voice input (Web Speech API, no server change). Still
unverified end-to-end against live Groq traffic after these specific fixes —
next test pass should confirm the silence protocol actually activates now
that the task-list-emptying bug is fixed.

Remaining:

- [x] Voice input — done 2026-07-13 via browser Web Speech API
      (`SpeechRecognition`/`webkitSpeechRecognition`), client-side only, no
      server change. Mic button (`#mic-btn` in `index.html`) is hidden
      entirely when the browser doesn't support it (no Firefox support as of
      this writing) — feature-detected in `initVoiceInput()` in `app.js`.
      Same secure-context requirement as the camera (HTTPS or localhost).
      Tap to talk, releases/auto-sends on the browser's own end-of-speech
      detection (`continuous: false`) — no second tap needed to submit.
- [ ] TTS output (text-only for now, by design — add once the text pipeline is
      trusted; Web Speech API first, since it's free and needs no server change)
- [ ] Adaptive poll backoff (currently a fixed 15s client poll; could back off to
      minutes when no timer is close to firing, bounded below Render's idle
      timeout so it doesn't reintroduce the spin-down problem)
- [ ] Render keep-alive pinger (only needed if unattended timers >15 min matter)
- [ ] Multi-timer progress display in the UI (`active` timers are already returned
      by `/v1/timers/check`, just not rendered anywhere yet)
- [ ] Streaming output (show thinking tokens as they arrive)
- [ ] Egocentric fine-tuning pipeline (Ego4D + Egocentric-1M data) — long-term
- [ ] Local GPU deployment (port from Colab/Render when hardware available)

---

## Fine-tuning (future)

Standard vision models were trained on third-person internet images.
Egocentric (first-person) footage from a phone camera has a different distribution:
hands in frame, close objects, shaky movement, indoor lighting.
Fine-tuning on egocentric data improves performance meaningfully.

Relevant datasets:
- Ego4D (Meta): 3670 hours, requires license agreement at ego4d.dev
- Egocentric-1M (Build AI): ~1M hours, Apache 2.0, released April 2026
- EPIC-Kitchens: 100 hours, kitchen-focused, densely annotated

Strategy: do not fine-tune from scratch. Identify failure cases on real camera
footage first, then fine-tune only on those failure patterns using targeted clips
from the datasets above.

---

## Non-Chinese model alternatives

The default Colab stack uses Qwen (Alibaba, Chinese); the active Groq model
(`qwen/qwen3.6-27b`) is also Qwen-family. If data residency or supply chain
concerns require non-Chinese models:

| Layer | Alternative | Notes |
|---|---|---|
| Vision | Gemma4 4B (Google) | Apache 2.0, runs on 16GB, good vision |
| Vision | Llama 4 Scout (Meta) | Image only, no video |
| Reasoning | Gemma4 27B (Google) | Apache 2.0, strong reasoning |
| Reasoning | Phi-4 (Microsoft) | MIT, 14B, strong reasoning for size |
| Full stack API | Gemini 2.0 Flash | Handles real video streams natively; already has a backend (`gemini_backend.py`) |
