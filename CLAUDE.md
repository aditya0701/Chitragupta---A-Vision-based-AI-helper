# Vision Chitragupta

A hands-free, camera-equipped voice assistant for hands-on tasks — cooking,
repairs, shopping, anything where you're working and can't look at a screen.
Named for the Hindu record keeper who observes, records, and reports.

It watches through a phone camera, reasons about what it sees, tracks everything
in flight (steps, substitutions, parallel work, timers), and speaks back. The
user is **listening, not reading** — that constraint drives most design choices.

> **`DECISIONS.md` holds the failure log and the reasoning behind every design
> choice here.** Read the relevant section before changing an area — most of the
> non-obvious code is scar tissue from a specific bug.

---

## Run it

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000     # from repo root
```

- Requires `TOOLS_ENABLED=true` in `server/.env` (defaults to `false`).
- **Avoid `--reload`** when editing `server/agent/*.py` — WatchFiles has served
  stale bytecode on Windows. Restart manually.
- Camera/mic need HTTPS or the literal hostname `localhost`. A bare LAN IP over
  HTTP fails the browser's secure-context check.
- Deployed on Render free tier (`render.yaml`) — public HTTPS, so any phone on
  any connection works. Sleeps after ~15 min of no inbound traffic.

---

## Architecture

```
Phone/browser ──► /v1/chat  or  /v1/chat/stream  (FastAPI)
                      │
                      ├─► [Stage 1] Groq qwen3.6-27b — VISION ONLY
                      │             image ──► short text description
                      │
                      ├─► [Stage 2] DeepSeek v4-flash — ALL reasoning + tools
                      │             (never sees pixels)
                      │
                      ├─► [Tools] native function calling
                      │   start_timer · cancel_timer · update_task_list
                      │   log_observation · request_camera · request_live_search
                      │   web_search · fetch_page · calculate · get_time
                      │
                      └─► text (+ think_blocks, tool_calls, goal_complete)
```

**Active backend is `hybrid`** — `BACKEND_MODE=hybrid` in `server/.env`. The split
exists to get reasoning off Groq's 8K tokens/minute cap; only the small image-only
prompt has to fit under it. See `DECISIONS.md` §1.1.

Other backends exist (`groq`, `gemini`, `openai`, `anthropic`, `colab`, `ollama`)
and are selected by `BACKEND_MODE` / `API_PROVIDER` via `backends/factory.py`.
They differ on two class flags: `SPLIT_VISION_REASONING` and
`SUPPORTS_NATIVE_TOOLS`.

### Why ReAct, not an orchestrator
The reasoning model orchestrates itself — it reads the scene, decides mid-thought
whether to call a tool, and routes to the right response type. The thinking chain
*is* the orchestration. Multi-agent designs were considered and rejected
(`DECISIONS.md` §9).

---

## Persistent state

Both under `server/data/` — gitignored, survives restarts **by design**.

| File | Module | What |
|---|---|---|
| `timers.json` | `agent/timers.py` | Wall-clock `start_time + duration`, not `asyncio.sleep` — survives a Render restart. Due-checks are pure arithmetic, zero LLM cost. |
| `document.json` | `agent/tasklist.py` | The task list. TodoWrite-style **full-list replace**. Items carry `status`, `note`, `observations[]`, `watch_for`. |

Both are injected into every reasoning prompt (`[Timers]`, `[Task list]`), so the
model acts on them without being reminded.

### Task item fields

```jsonc
{
  "id": "a1b2c3d4",
  "content": "Check broth base is compliant",   // exact key; aliases accepted
  "status": "in_progress",                      // pending|in_progress|completed|skipped
  "note": "used tofu instead of paneer",        // substitutions
  "observations": ["..."],                      // max 5, oldest dropped
  "watch_for": "Read the ingredient list and list any of: beef, gelatin, lard."
}
```

**`watch_for`** is the reasoning model's own brief to the vision stage. It cannot
see the camera; a separate model looks on its behalf and reports only what it was
asked for. Briefs must request **observations, not judgement** — the reasoning
model supplies the critique. See `DECISIONS.md` §6.3.

---

## Live Watch

Camera samples on an interval with a **client-side perceptual diff gate** — if the
scene hasn't meaningfully changed, no request leaves the browser. That gate is the
main cost control.

- Silent ticks: `LIVE_FRAME_DIM = 640`. Typed question riding along: `MAX_FRAME_DIM = 1024`.
- A tick firing mid-request is buffered (`pendingLiveFrame`), not dropped.
- **Silence protocol** — on a live tick with an active goal, if nothing is new the
  entire reply must be exactly `[SILENT]`, stripped server-side. Direct user turns
  are never allowed to go silent.
- `log_observation` runs on every relevant tick — text facts accumulate so "where
  is X" can be answered from history, not just the current frame.

Full step-by-step pipeline: `DECISIONS.md` §6.4.

---

## Hard rules

Each of these cost a real debugging session. Details in `DECISIONS.md`.

| Rule | Why |
|---|---|
| **Bump `CACHE_NAME` in `sw.js`** whenever `index.html`/`app.js`/`style.css` change | Cache-first SW otherwise serves a stale shell forever. Currently `v16`. §8.1 |
| **Mirror changes across `_process_locked` and `_process_stream_locked`** | They duplicate tool availability, recovery and response shape. A fix in one is a bug in the other. §5.2 |
| **Mirror `app.js` ↔ `debug.js`** | Camera and frame logic exist in both. §8.2 |
| **One flag, one consequence** | `found` drove three unrelated outcomes and broke the camera. §4.4 |
| **Never add a pre-filter whose "no" looks like silence** | Cost a whole class of silently dropped frames. §6.2 |
| **Don't add `check`/`start`/`look`/`find` to `_ACTION_CUE_RE`** | They appear inside the non-answers it exists to catch. §7.2 |
| **Use `_locked` variants internally** | `_lock` is not reentrant. §4.3 |
| **Image cost scales with resolution, not JPEG quality** | Quality is not a lever. §1.2 |

---

## Constraints

- **Groq: 8,000 tokens/min and 200,000 tokens/day.** The daily cap is what kills
  long sessions — one 47-minute run died at ~198k. In the hybrid, Groq does vision
  only, so **image tokens are essentially the whole budget**.
  `deepseek_backend.vision()` logs `Groq vision usage:` per frame.
- **Render sleeps after ~15 min** with no *inbound* traffic. Frontend polling keeps
  it warm during use; a timer firing while the phone is closed is delayed.
- **`TOOLS_ENABLED` defaults to `false`.** Nothing above works until it's `true`.

---

## Layout

```
CLAUDE.md          this file — working reference
DECISIONS.md       failure log + design reasoning
render.yaml
server/
├── main.py                FastAPI app, /v1/* routes
├── config.py              settings from .env
├── agent/
│   ├── agent.py           ChitraguptAgent — prompt building, tool execution, recovery
│   ├── __init__.py        Tool/ToolRegistry, built-in tools, ConversationMemory
│   ├── timers.py          persisted wall-clock timers
│   └── tasklist.py        persisted task document
├── backends/
│   ├── __init__.py        VisionBackend ABC, VisionResponse, should_think()
│   ├── deepseek_backend.py   ACTIVE — Groq vision + DeepSeek reasoning
│   ├── groq_backend.py, gemini_, openai_, anthropic_, colab.py, ollama_
│   └── factory.py         get_backend()
├── live/                  PARALLEL v2 system — see below
└── static/                no build step
    ├── index.html, app.js, style.css      main UI
    ├── debug.html, debug.js               raw pipeline view
    ├── live.html, live.js                 v2 UI
    └── sw.js, manifest.json               PWA
```

---

## The v2 system (`server/live/`)

A **separate, parallel** workflow — `/v2/*` API, `/live` page, own state
(`data/live/worlddoc.json`). Shares only the backend classes and `Tool`/
`ToolRegistry`. Nothing in `server/live/` is imported by v1.

**Design inversion:** the world document is primary state and speech is a
side-effect. Ticks update the doc continuously; a zero-cost arithmetic trigger
engine decides when to wake the reasoning model.

- `worlddoc.py` — tasks, expectations, durable environment facts, compacted
  narrative, raw recent captions. Section order is stability-first for prefix-cache hits.
- `triggers.py` — overdue expectations + stale tasks, checked by arithmetic. A
  politeness budget gates unprompted speech.
- `vision.py` — passes the previous caption back as text so the model describes
  *change*, not an isolated snapshot.
- `compaction.py` — span-preserving summarisation when `recent` overflows.

**Status: verified by a 27-check fake-backend smoke test, never run against live
traffic.** v1 is what actually gets used. Building a feature in v2 means debugging
two new things at once.

---

## Current state

Working and exercised in real sessions: the hybrid pipeline, native tool calling,
timers, task tracking, live watching with silence, voice in (Web Speech
`SpeechRecognition`) and out (`speechSynthesis`), PWA install.

Recent work (see `git log`): the `found`/`alert` split, `watch_for`, the
`request_camera` double-fire fix, camera-ready frame capture, live-tick resolution
cut, generalisation away from cooking-only wording, and `cancel_timer`.

**Next up, in order:** resolution tiers + a fine-detail inspection tool (needed
before shopping/label reading is usable), then adaptive poll backoff. Full list
with rationale: `DECISIONS.md` §10.

There is no automated test suite — verification is ad-hoc harnesses plus reading
exported session wire logs. `DECISIONS.md` "Testing notes" explains why the wire
log is the highest-value artifact.
