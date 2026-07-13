# Handoff — 2026-07-13 session

Read this first if you're picking this project up cold. It orients you
fast; `Claude.md` has the full architectural detail for anything you need
to go deeper on. Don't duplicate content between the two — update
`Claude.md` for anything architectural, update this file only for
session/status framing (what's tested, what's next, what's blocking).

## What happened this session, in one paragraph

Started from a live-tested transcript showing three failures: Live Watch
repeating the same scene description every tick, a typed question about an
object's location answered blind (no image ever attached), and rate-limit
errors. Root-caused and fixed across ~10 passes: added a goal-directed
observation log + silence protocol so the model stops narrating and starts
checking against an actual goal; added `request_camera` (pull one frame on
demand) and `request_live_search` (model-initiated continuous watching,
scoped to finding one thing); fixed a real concurrency bug (no lock existed
across live ticks / chat / timer polls); switched Groq tool-calling from
regex-parsed free text to native function-calling after a field-name-drift
bug silently emptied the task list; fixed two truncation-related bugs where
the model's reasoning ran past `max_tokens` and either leaked raw text or
left the answer blank; added voice input (Web Speech API); decoupled the
camera stream from Live Watch's polling loop so it can be used on-demand;
fixed a stale service-worker cache that was hiding all of today's frontend
changes from anyone who'd visited the app before today.

## Current state — what's confirmed vs. only unit-tested

**Confirmed working against real Groq traffic** (from an actual test
transcript, not just unit tests): task list creation with correct
`content` keys, sequential ingredient-finding across manually-uploaded
images, no crashes, no duplicate memory entries, no repeated-narration
spam in that run.

**Implemented and unit-tested (fake backend), NOT yet confirmed live:**
- `request_live_search` — the whole goal-registration → `needs_live_search`
  → client auto-switches to Live Watch → subsequent ticks check the goal
  chain has never been run against real Groq. **This is the natural next
  test.**
- The two truncation-recovery paths (conclude-from-reasoning /
  fresh-retry) — verified with synthetic truncated responses, not a real
  truncation from Groq since the last fix.
- The `request_camera` guidance-text nudge (telling the model to actually
  call the tool instead of explaining the UI) — prompt-level nudge, no
  guarantee it changes model behavior. Worth specifically watching.

## Known open bugs (not yet fixed)

1. **Input-hijack race, `app.js` `sendLiveFrame()` (~line 550).** It reads
   and clears `#prompt-input` on its own interval tick, independent of the
   Send button. If you type while Live Watch is polling, a tick can grab a
   partial/leftover value before you finish. This was the root cause of
   the "Ant to" fragment / non-answer confusion from an early transcript
   this session. Diagnosed, never patched — deprioritized in favor of
   backend reliability work. **Should probably be next after the live
   `request_live_search` test**, since it directly affects that mode.
2. **No auto-stop when a `request_live_search` target is found.** Live
   Watch keeps polling indefinitely after success until the user manually
   leaves the tab or toggles the camera off. Ties to the deferred
   `start_watch`-style idle-cost item below.
3. **Partial-evidence task completion.** A compound task-list item (e.g.
   "Gather Ingredients" covering 10 things) can get marked `completed`
   from a single frame showing 2 of them. Would need per-ingredient
   sub-items to actually fix — separate scope, not started.

## Deferred / lower priority (from `Claude.md`'s TODO)

TTS output, streaming output, multi-timer progress UI, adaptive timer-poll
backoff, Render keep-alive pinger, no pruning of a *completed* item's
observation log, no retry/backoff on a repeated camera-unavailable
response. None of these were touched this session; see `Claude.md`'s
`## TODO` section for the full list.

## Suggested next steps, in order

1. **Live-test `request_live_search`** — say "help me find X" from Chat &
   Image with no image attached, confirm it registers the goal and
   switches into Live Watch automatically.
2. Fix the input-hijack race (#1 above) before doing extended live-mode
   testing, since it'll otherwise confuse results.
3. Decide whether to add the auto-stop-on-found behavior (#2) now or treat
   it as its own follow-up.

## Where to look for detail

`Claude.md`, in roughly the order you'll need it:
- `## Agentic tools & persistent state` — timers, task list, observation
  log, all the tool-calling mechanics.
- `### Live Watch pipeline, step by step` (under `## UI modes`) — the full
  client-to-server-and-back walkthrough, written this session specifically
  to consolidate what was previously scattered across multiple pass-notes.
- The dated pass-notes throughout (search `2026-07-13`) — each documents
  one bug found (often from a real test transcript) and the specific fix,
  in the order they happened. Read chronologically if you want the full
  reasoning trail; skim just the bold headers if you want the summary.
- `## TODO` at the bottom — the authoritative running list.

## Repo state

Branch `main`, all work pushed and deployed to Render (auto-deploys on
push to `main`). Latest commit as of this handoff: `f1d1ca6`. No
uncommitted work.
