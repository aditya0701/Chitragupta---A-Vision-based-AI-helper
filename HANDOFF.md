# Handoff — 2026-08-10

Read this first if you're picking the project up cold. `CLAUDE.md` is the v2
working reference; `DECISIONS.md` is the v2 failure log. **Don't duplicate those
here** — this file is session/status framing only: what's confirmed, what's
next, what's blocking.

**Docs were restructured today.** v2 is now the primary system in both docs; v1
is archived under `docs/v1/` and is not being developed. Code comments citing
`DECISIONS.md §6.3` / `DECISIONS.md 4.4` point at `docs/v1/DECISIONS.md` — those
numbers were assigned before v2 had a log, and were deliberately not renumbered.

**Repo state:** `main` is ahead of `origin/main` — push before handing off.
`tests/run_all.py` → **14/14 passing.**

---

## v2 has now run live

A ~18-minute chicken curry session, 2026-08-10, exported as
`chitragupt-live-20260810-1342.md` (56 entries). **This is the first real
traffic v2 has seen**, and it answers the three questions the previous handoff
was blocking on.

### Confirmed working

| Thing | Evidence from the export |
|---|---|
| **Propose-then-commit** | Plan held unwritten across several minutes and many ticks. Raised unprompted: *"I'm still waiting to hear if that style works for you"*. `commit_plan` fired on "yes this style works for me", together with `update_tasks` + `set_vision_focus` |
| **Plan spoken, not just written** | 6 steps read out in the same reply as the commit — the user never had to ask what was in the document |
| **Caption reuse** | `capture: user turn — scene unchanged, reusing the last caption` |
| **Blank camera** | Reported honestly and usefully: *"my view is completely black (looks like a lens cap)… carry me with you and the moment the view clears I'll look"* |
| **`[URGENT]`** | Fired on a motion-blurred frame in a tight passage: *"Slow down—path is tight by the door and shelving"* |
| **Compaction** | Narrative entry covering `13:24:56–13:36:58` — time span preserved, not just facts |
| **Durable spatial memory** | 8 environment facts by export, including the fridge shelf the chicken came off |
| **Mid-plan adjustment** | Breasts-not-thighs propagated into two task notes with the reasoning attached (*"BREASTS — keep short to avoid drying out"*) |
| **Honest absence** | Asked twice about onions, said plainly it hadn't seen any rather than guessing |

### Found broken

**1. The session died on a Groq rate limit** — *fixed, see "Next" item 1.*
~18 minutes in:

> `429 — Rate limit reached for model qwen/qwen3.6-27b … TPD: Limit 200000, Used 199109`

**v2 should never touch Groq.** `server/.env` (untouched since 31 July) and
`render.yaml` both set `LIVE_BACKEND_MODE=deepinfra`, and the factory routes
that cleanly to `DeepInfraHybridBackend`. A missing `DEEPINFRA_API_KEY` can't
explain it either — that raises at construction and would fail every request.
The only route from `/v2/chat` to Groq is an un-overridden `DeepSeekBackend`,
i.e. the process resolved to `hybrid`, so it was almost certainly a **stale
server predating the config**.

`get_live_agent()` already logs `Initialized live agent with backend mode:` at
startup. **Check that line before the next session** — it settles this in one
glance. `DECISIONS.md` §5.3.

**2. A dangling reply shipped.** The turn ended:

> *"Let me also make a note of what's on the menu so it stays consistent:"*

`_repair_dangling_plan` exists for exactly this and did not fire: it requires a
tool from `_PLAN_TOOLS` (`propose_plan` / `update_tasks` / `mark_task`) in the
results, and this turn's only tool was `log_environment`. The trailing-colon
regex matched; the tool guard rejected it. See
[agent.py:116-146](server/live/agent.py#L116-L146).

**3. Flat frames are still captioned at full price.** Detection works, but the
blackout at 11:36–11:38 was described by the vision model five times. Detected,
reported, still billed. `DECISIONS.md` §7.2.

**4. `fine` detail was never once used.** Every caption in the session is
`(coarse)`, including reading a patent binder and searching a fridge. The tier
works in the harnesses but the model never chose it live — worth watching
whether `set_vision_focus(detail="fine")` is reachable in practice or whether
the prompt is steering away from it.

---

## Next, in order

### 1. The backend bug — **DONE**

v2 can no longer end up on Groq by accident. `LIVE_BACKEND_MODE` now defaults
to `deepinfra` (it defaulted to the Groq path, so the default nobody sets was
the one config that cannot work), backends declare `VISION_PROVIDER` so the
question is answerable at all, and `_check_vision_provider()` raises unless
`LIVE_ALLOW_GROQ_VISION=true`. `main.py` warms the agent at boot so the check
runs before anyone props a phone up — without re-raising, since v1 shares the
process and must keep serving.

Verified end to end: with a deliberately bad `LIVE_BACKEND_MODE=hybrid`, `/health`
still returns 200 while `/v2/tick` returns the refusal and its fix. Guarded by
`tests/t_live_backend.py`. `DECISIONS.md` §5.3.

**The startup line to check** is now:

```
Initialized live agent | backend mode: deepinfra | VISION ON: deepinfra | reasoning: deepseek-v4-flash
```

### 2. The other two bugs from the export

Both still open, both small: widen the dangling-reply guard beyond
`_PLAN_TOOLS` (finding 2), and skip the vision call entirely on a frame already
known to be flat (finding 3).

### 3. `/v2/chat/stream`

SSE endpoint streaming the final reasoning call, client flushing to `speak()` on
sentence boundaries. This was queued behind the cook test, which has now
happened — it is unblocked.

**Design constraint, committed to:** a single `_chat_turn` with an optional emit
callback, **not** a second parallel function. v1's
`_process_locked`/`_process_stream_locked` divergence
(`docs/v1/DECISIONS.md` §5.2) is precisely what this must not inherit.

---

## Longer-standing, unchanged

Full reasoning in `DECISIONS.md` §9.

- **Adaptive tick backoff** — nothing throttles a 20-minute simmer.
- **The diff gate dies while walking** — every frame differs, so it skips
  nothing. The export shows this clearly during the fridge/pantry search.
- **Partial-evidence task completion** — a compound step ("Prep: dice onions,
  dice tomatoes, make paste, cut chicken") can be marked done from a frame
  showing one of four. Needs sub-items.
- **Per-user timezone** — `DEFAULT_TIMEZONE` is one server-wide setting.
- **v1**: `[Timers]` injection, `start_timer` and `cancel_timer` were never
  exercised live. Frozen along with the rest of v1.

---

## Working notes

- **Restart the server manually** — not `--reload` (stale bytecode on Windows).
  Given finding 1, this matters more than it looks.
- **`/` now serves v2**; `/live` stays as an alias, v1's UI moved to `/v1`, and
  v1's API is untouched on `/v1/*`. Deployed at
  <https://chitragupta-k6ek.onrender.com/>.
- **v2 needs no `CACHE_NAME` bump for ordinary work.** `/`, `/live`,
  `/static/live*` and `/v2/*` are excluded from the service worker outright
  ([sw.js](server/static/sw.js)). `sw.js` went to `v20` once, for the `/` handover
  — `/` was cached cache-first as part of v1's shell, so returning browsers would
  have been served v1 at the new default address. v1's shell is cached at `/v1`
  now. The manifest link moved to `live.html` at the same time, so the default
  page is still installable on a phone.
- **No test suite.** `tests/run_all.py` is 13 ad-hoc harnesses. They cover what's
  checkable without a live session — nothing about *judgement*. Every finding in
  the "broken" list above came from reading the export, not from a harness.
- **The export is the highest-value artifact.** Every vision prompt and answer,
  silent ticks included, plus the world document as it stood. Read it top to
  bottom; no single turn shows a flip-flop, only the sequence does.
- **Ask for server logs alongside any export.** `Initialized live agent with
  backend mode:` and `DeepInfra vision usage:` (the per-frame bill — in this
  split the vision call is the only image cost).
