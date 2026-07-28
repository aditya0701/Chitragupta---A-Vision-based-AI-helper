# Handoff — 2026-07-28 session

Read this first if you're picking the project up cold. `CLAUDE.md` is the
working reference for how the system behaves; `DECISIONS.md` is the failure log
and the reasoning behind it. **Don't duplicate those here** — this file is
session/status framing only: what's confirmed, what's next, what's blocking.

**Repo is clean and pushed** — `origin/main` @ `7424a4a`. Nothing in flight.

---

## What happened this session

13 commits, then one real ~40-minute session on a phone (lentil soup, then
hot-pot shopping). Work fell into three groups:

**Fixed from the previous transcript** — the "I updated the list" non-answer
(persona rewritten into a role brief + a deterministic backstop), the
`request_camera` double-fire (tool filtering was missing on the streaming path),
and camera cold-start returning frameless captures.

**Found while fixing those** — two bugs nobody had reported. The streaming path
never ran the vision stage at all, so *every* typed turn carrying an image
discarded it silently while telling the model an image was attached; that made
`request_camera` a no-op on this backend since its entire purpose is delivering
a frame. And the "scene unchanged" short-circuit had no `is_live_frame` guard,
so a typed question about a static scene got the monitoring blurb instead of an
answer.

**New capability** — `watch_for` (the reasoning model writes its own brief to
the vision stage instead of us inferring one from task text), a coarse/fine
detail tier so the camera can read labels, `cancel_timer`, a `[Timers]` block so
the model can see what's running, generalisation away from cooking-only wording,
and every Qwen prompt/answer now recorded in the export.

Docs were also split: `CLAUDE.md` slimmed to a working reference, `DECISIONS.md`
created as the failure log.

---

## Confirmed working against live traffic

All of it held up. Nothing regressed. Evidence from the session export:

| Thing | Evidence |
|---|---|
| **Fine-detail label reading** | Model set `detail:"fine"` itself, client logged `🔍 frame detail → fine (1024px)`, Qwen read `"Lamb 55%," "Water," "Sugar," "Carrageenan"` off a package |
| **`watch_for` briefs** | Model wrote *"Read every German label clearly… Flag if you see: Rindfleisch, Rind, Rinder-, Kalb…"* unprompted — and fact-shaped, not judgement-shaped |
| **`alert` / `found` split** | 3× `alert:true`, 1× `found:true`. Camera stayed on through ordinary steps, closed only on a real find-goal |
| **Anti-false-absence rule** | On a blurred frame: *"I cannot assign a safe or unsafe verdict"* rather than a confident false negative — on a dietary-restriction question |
| **Previous-caption grounding** | Captions describe change against the prior frame instead of re-describing in isolation |
| **Vision on the streaming path** | Vision blocks now appear on typed turns |
| **Persona / generalisation** | Opens with *"Cooking, fixing something, shopping, or something else?"* |
| **Concrete steps** | *"let's rinse the lentils and chop the onions"*, not *"I updated the list"* |

**Not exercised:** `cancel_timer`, `start_timer`, and the `[Timers]` prompt
injection — no timers came up. Needs a session with timers in it.

---

## Next, in order

### 1. Bound the cost of `detail: "fine"` — it caused a 429

Session died at **198,310 / 200,000 TPD**, same wall as last time.

A fine frame costs `Requested 3424` tokens against roughly 1400 for coarse — the
predicted ~2.4x. But:

```
11:26:40  🔍 frame detail → fine (live ticks now 1024px)
          ...never switches back
```

Set once, never reverted. Cost was bounded by **scope** — "completing the step
ends it" — and the step stayed `in_progress` for the rest of the session, so
every later tick ran at full resolution.

**The lesson: a cost control that depends on the model's judgement does not
hold.** The schema explicitly tells it to set detail back to `coarse` when the
close look is done. It didn't.

Needs a hard bound alongside the scope bound:

- count fine frames per item and auto-revert to `coarse` after N (5–10?) — the
  counter belongs on the item so it survives a restart;
- or revert once the `watch_for` question has been answered (harder — needs a
  notion of "answered");
- or a session-wide fine-frame budget.

Files: `server/agent/tasklist.py` (`active_detail()`, ~L167),
`server/agent/agent.py` (`_run_vision_stage`). Background: `DECISIONS.md` §6.4.

### 2. Let a user correction retract an observation

Worst behaviour in the session, and a design gap rather than a bug.

The model found "toor dal" in a plastic bag and logged `found:true`. The user
corrected it — *"the thing in the plastic bag is black eyed beans not toor
dal"*. Nothing retracted it. `Find toor dal` stayed `completed`. The wrong note
kept riding along in `[Task list]`, and the model confabulated for four turns:

> *"the toor dal bag is on an upper shelf, open and upright, with a black loaf
> pan sitting behind it"*

The user challenged it twice (*"When did you told me about the dal on the upper
shelf?"*), it apologised — and then **repeated the claim**.

`log_observation` is append-only. There is no `retract_observation`, and no way
to un-complete a wrongly-completed item. A false observation that survives an
explicit correction is worse than no observation at all.

Needs a way to invalidate a specific observation, a way to move `completed` back
to `in_progress`, and prompt guidance to use both when the user says "no,
that's wrong".

Files: `server/agent/tasklist.py` (`add_observation`, ~L208),
`server/agent/__init__.py` (tool registration), `server/agent/agent.py`
(guidance).

Decide at the same time: the phantom "black loaf pan" persisted across frames,
possibly a side effect of the §6.5 grounding rule telling the model not to
contradict the previous caption. Consider "…unless the user corrected you."

### 3. Silence narration leaked once

Spoken aloud on a watch tick:

> *"Nothing has visibly changed… **I'll stay quiet until there's something new to
> report.**"*

`_SILENCE_NARRATION_RE` ([agent.py:36-43](server/agent/agent.py#L36-L43)) exists
for exactly this and missed two near-variants:

- `"stay quiet"` — the pattern covers `stay|remain|keep|be` + **silent** only
- `"Nothing has visibly changed"` — covers `nothing` + `relevant|new|of note|worth …`, not `nothing has … changed`

At ~281 chars it was under the 300 cap, so only the regex failed. One occurrence
in 42 ticks. Cheap — but read `DECISIONS.md` §7.2 first, widening these regexes
has bitten before.

### 4. `get_time` ignores its own `timezone` parameter

[`tool_get_time`](server/agent/__init__.py#L155) is a stub that always returns
UTC. The model called it twice with `Asia/Kolkata`, got UTC both times, then
reasoned out loud about timezone offsets from a value it had been told was
something else. Small and self-contained.

### 5. Input-hijack race in `sendLiveFrame()` *(carried over, never fixed)*

`sendLiveFrame()` reads and clears `#prompt-input` on its own interval tick,
independent of the Send button. Type while Live Watch is polling and a tick can
grab a partial value mid-keystroke. Diagnosed on 2026-07-13, still present, and
it directly affects the mode we now use most.

Files: `server/static/app.js` (`sendLiveFrame`, the `typedPrompt` read), mirror
in `debug.js`.

### 6. `web_search` timed out

10s against DuckDuckGo's HTML endpoint
([__init__.py:88](server/agent/__init__.py#L88), L125). Failed once; the model
recovered gracefully from its own knowledge, which is right — but the tool is
unreliable enough to revisit.

### 7. Reply length for a listening user

Early replies were essay-length markdown — bold, bullets, headers — for someone
who can't see the screen. Later ones were much better:

> *"Hey! That surimi package you're looking at — it's all fish meat, no beef in
> sight, so you're totally safe."*

That's the target voice. The persona already says the user is listening; it may
need an explicit length ceiling for non-planning turns.

---

## Longer-standing, unchanged

Full ranked list with rationale in `DECISIONS.md` §10:

- **Adaptive poll backoff** — nothing throttles a 20-minute simmer.
- **The diff gate dies while walking** — every frame differs so it skips
  nothing; shopping burns TPD far faster than cooking.
- **Context loss** — root cause still unknown, now instrumented (§7.7). Needs a
  recurrence; the `turn context:` log line will answer it immediately.
- **Attach the live frame in `sendMessage`** — removes the `request_camera`
  round trip from the common case.
- **One-shot `inspect_detail`** — close look with no standing goal.
- **`request_live_search` misuse** — flagged in the original transcript
  analysis, still not written up in `DECISIONS.md`. It gets invoked for things
  that aren't "find one object".
- **Partial-evidence task completion** *(carried from 2026-07-13)* — a compound
  item ("Gather ingredients", covering ten things) can be marked `completed`
  from a frame showing two. Needs sub-items.

---

## Working notes

- **No test suite.** Verification is ad-hoc harnesses plus reading exported
  sessions. Five harnesses were written for this work (flag split, streaming
  vision, detail tier, flip-flops, export) and live in the scratchpad, not the
  repo. Rewrite as needed — they're cheap, and the export-testing one runs the
  real `app.js` in a stubbed DOM rather than reimplementing it.
- **The export is the highest-value artifact.** It now carries every Qwen prompt
  and answer, silent ticks included, both inline per turn and as a chronological
  section. Reading that section top to bottom is how the flip-flops were found —
  no single turn shows them, only the sequence does.
- **Ask for server logs alongside any export.** Two lines matter:
  `turn context:` (history turns, task items, whether `[Task list]` reached the
  prompt) and `Groq vision usage:` (per-frame image cost).
- Restart the server manually — **not** `--reload` (stale bytecode on Windows).
- Bump `CACHE_NAME` in `sw.js` for any `index.html`/`app.js`/`style.css` change.
  Currently `v18`.
