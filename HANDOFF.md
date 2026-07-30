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

### 1. Bound the cost of `detail: "fine"` — **DONE**

Session died at **198,310 / 200,000 TPD**. `detail:"fine"` was set once and
never reverted, so every later tick ran at full resolution.

Fixed with an arithmetic bound beside the scope bound:
`MAX_FINE_FRAMES_PER_ITEM = 8`, charged per item by `record_fine_frame()` from
the vision stage, counter stored on the item so a restart doesn't refill it.
On exhaustion the item is written back to `detail:"coarse"` so the prompt stops
claiming a close look that isn't happening.

**The trap, which the test caught and the obvious implementation walks into:**
the auto-revert writes `coarse`, and `update_task_list` resends every item every
call — so the next full replace still saying `fine` looked like a deliberate
`coarse`→`fine` transition and refilled the budget. Exhaustion is now recorded
explicitly (`fine_budget_spent`) and only the model itself sending `coarse`
clears it. A legitimate second close look is a deliberate two-step: stand down
to `coarse`, then request `fine` again.

`DECISIONS.md` §6.4.

### 2. Let a user correction retract an observation — **DONE**

The toor dal / black-eyed-beans failure: a wrong `found:true` observation that
nothing could remove, kept riding along in `[Task list]`, confabulated from for
four turns, challenged twice, apologised for, then repeated.

Fixed with `retract_observation(item, note_match, reopen=False)` — removes every
observation containing `note_match` (substring, not index: the model can quote a
fragment far more reliably than it can count positions), and with `reopen=true`
moves a wrongly-`completed` item back to `in_progress` and clears a matching
`note`. Prompt guidance tells the model that apologising in conversation does
*not* undo a logged note, and to believe the user over its own earlier note.

The §6.5 interaction was real and is fixed too: the previous-caption grounding
now ends with *"consistency with an earlier mistake is not consistency"*, so it
stops reading as an instruction to keep re-asserting a wrong caption.

Verified by replaying the exact transcript scenario. `DECISIONS.md` §6.6.

**Not yet exercised against live traffic** — both need a real session.

### 3. Silence narration leaked once — **DONE**

> *"Nothing has visibly changed… **I'll stay quiet until there's something new to
> report.**"*

Both near-variants added (`quiet` alongside `silent`; a subject-first
`nothing … changed` branch, since the existing no-change branch was
`scene|frame|everything` + `remains`).

**But widening alone would have been the wrong fix**, and §7.2 is only half the
story. Widening `_ACTION_CUE_RE` is safe because it makes that detector *less*
likely to fire. This regex runs the other way — widening makes suppression
*more* likely, and a false positive means the user never hears something real.
The counter-example the old check couldn't separate from the leak:

> *"Nothing has changed with the heat, but the onions are starting to brown —
> give them a stir."*

So `_is_silent_live_reply` gained the third condition it was missing, matching
`_is_list_meta_nonanswer`'s shape: short, **and** no-change vocabulary, **and**
nothing substantive — an action imperative or a digit. Bias is deliberately
toward leaking a redundant "nothing changed" over swallowing an update.

`_ACTION_CUE_RE` itself was **not** touched (the §7.2 trap). 27-case harness
covers both directions. `DECISIONS.md` §7.8.

### 4. `get_time` ignores its own `timezone` parameter — **DONE**

It was a stub that always returned UTC. The model called it twice with
`Asia/Kolkata`, got UTC both times, then reasoned out loud about offsets from a
value it had been told was something else — the same shape as §3.6: the tool
answered a different question than the one asked, without saying so.

Now backed by `zoneinfo`, defaulting to **`DEFAULT_TIMEZONE`
(`Europe/Berlin`)** since "what time is it" means local time. Every reply names
the zone it actually used, and an unrecognised zone says it fell back instead of
quietly returning UTC.

**`Europe/Berlin`, not `CEST`** — Berlin is CET in winter and CEST in summer, so
an abbreviation is wrong half the year. Verified switching correctly across the
October DST boundary. (`CEST` passed as an argument isn't a valid IANA name; it
falls back with a note, landing on the right zone anyway.)

Output is phrased for speech first — *"16:10 on Wednesday 29 July 2026 —
Europe/Berlin (CEST)"* — with the ISO form trailing for arithmetic, since a bare
ISO timestamp read aloud by TTS is unusable.

`tzdata` added to `requirements.txt`: `zoneinfo` reads the OS tz database, which
Windows lacks and slim Linux images often omit, so without it this silently
degrades to UTC on some hosts.

**Per-user timezone is the eventual shape** — this is one server-wide setting
standing in until there's somewhere to store per-user preferences at all.

### 5. Input-hijack race in `sendLiveFrame()` — **DONE** *(carried from 2026-07-13)*

The interval read and cleared `#prompt-input` itself, so a tick landing
mid-keystroke sent a half-typed question and blanked the box.

Fixed with `queuedLivePrompt` — set only by an explicit commit (Send / Enter),
consumed by `sendLiveFrame` so it can't be resent. **The interval no longer
touches the textarea at all.**

**Behaviour change worth knowing:** a commit during Live Watch now goes out on
the live path **with a frame**, immediately, instead of `sendMessage()`'s
frameless `/v1/chat/stream`. Previously the same question reached the model two
different ways depending on whether you pressed Enter or waited for a tick, and
only the accidental path could see. Trade-off: typed questions during Live Watch
no longer stream — but they never did when a tick picked them up, so this is
consistency, not a loss. Partial down payment on "attach the live frame in
`sendMessage`" below.

Mirrored in `debug.js` (§8.3), `CACHE_NAME` bumped to **v19** (§8.1).

Verified with a Node harness driving the real `app.js`/`debug.js` in a stubbed
DOM — 14/14 both files, and **9 failures against the pre-fix file**, so the
harness demonstrably catches the bug. See `DECISIONS.md` §8.5 for the
`vm.runInContext` trap that made the first run of that harness meaningless.

### 6. `web_search` timed out — **DONE**

Worse than a timeout once opened up. DuckDuckGo serves its bot CAPTCHA with
**HTTP 202**, so `raise_for_status()` passed, the parse found zero results, and
the tool returned `No web search results found for "X"` — **a hard block
reported to the model as an authoritative absence**, on exactly the
does-this-contain-beef questions where that is most dangerous.

Fixed: `SearchBlocked` keeps "blocked" and "genuinely empty" as visibly
different tool results, and a provider chain replaces the single endpoint —
Brave (if `BRAVE_API_KEY` is set) → Mojeek → DDG lite → DDG Instant Answer.
Also found and fixed: tools ran on the event loop, so a slow search stalled
every live tick.

Full write-up including the counter-intuitive measurements (a spoofed browser
UA gets *more* CAPTCHAs than an honest bot UA; the "slow endpoint" was a cold
TLS handshake): `DECISIONS.md` §3.6 and §3.7.

**Optional:** set `BRAVE_API_KEY` (2,000 searches/month free) to put a provider
with an actual contract at the front of the chain. Everything works without it.

### 8. The apology loop — **DONE** *(reported 2026-07-29, not in the original list)*

Same session as item 2, different failure: after being corrected the model
apologised, was challenged, apologised again, and never answered — while the
vision stage was handing it good fresh information.

Mechanical cause of "never confirmed it": `found=true` marked the item
**completed**, and every camera path filters on `status == "in_progress"`, so
the search dropped out of the vision briefing and the camera closed. It had
stopped looking. Item 2's `reopen=true` restores the goal and its brief; item
2's retraction stops the wrong note being re-injected every turn.

What was left uncovered — what to do *after* apologising — is now:

- **Persona:** believe the user over your own note, say sorry once and briefly,
  then answer; never let an apology be the whole reply.
- **Backstop** keyed on **repetition, not vocabulary** — *"Sorry, those are
  black eyed beans, not toor dal"* is an apology *and* the right answer, so no
  word-list can separate them. Fires from the **second** consecutive apology;
  the first is left alone. A real answer resets the streak.

Both paths wired (§5.2). 17-case harness. `DECISIONS.md` §7.2b — including two
regex branches that matched nothing and passed the tests anyway.

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
