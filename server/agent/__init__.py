"""Agentic core — tool registry, memory, and the main agent loop."""

from __future__ import annotations
import json
import re
from typing import Any, Callable, Optional


class Tool:
    """A tool the agent can invoke."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., str],
        parameters: dict[str, dict],
        needs_followup: bool = True,
        blocking: bool = False,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters
        # True for tools that make a synchronous network call. Every tool runs
        # via `tool.fn(...)` from inside an async handler, so a slow one stalls
        # the whole single-worker event loop — during Live Watch that means the
        # 4s camera ticks, timer checks and any other request all queue behind
        # it. web_search could hold that for its full per-provider timeout
        # times four providers. The executors in agent.py hand these to
        # asyncio.to_thread instead of calling them inline.
        #
        # Deliberately opt-in rather than applied to every tool: the task-list
        # and timer tools mutate state under a non-reentrant lock (DECISIONS.md
        # §4.3), and moving those off the event-loop thread would change the
        # concurrency assumptions they were written under. Only the two
        # stateless network tools are flagged.
        self.blocking = blocking
        # Whether a call to this tool warrants a second Groq call to weave its
        # result into the reply. True for tools that surface new information
        # (web_search, fetch_page) the model hasn't seen yet. False for tools
        # that are pure side effects with a self-explanatory result
        # (start_timer, update_task_list) — the model's own visible text
        # around the tool call already says what it needs to; regenerating it
        # would just be a second paid call to restate the same thing.
        self.needs_followup = needs_followup

    def to_openai_tool(self) -> dict:
        # "required" is a per-parameter flag in our own Tool definitions
        # (convenient for the prompt-text renderer in agent.py), but real
        # JSON Schema wants it as a sibling list of names, not a property of
        # each property — leaving it inline was harmless noise for the
        # never-used old to_openai_tools() path, but native tool calling
        # (added 2026-07-13) actually sends this schema to the API, so it
        # needs to be valid.
        properties = {k: {kk: vv for kk, vv in v.items() if kk != "required"}
                      for k, v in self.parameters.items()}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [k for k, v in self.parameters.items() if v.get("required")],
                },
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]


# ─── Built-in tools ───────────────────────────────────────────────────────────

# Module-level rather than the per-function imports the rest of this file uses:
# the search chain below has four providers that all need it, and bs4 is a hard
# dependency already so there is nothing to defer.
from bs4 import BeautifulSoup

# A plain self-identifying UA, deliberately NOT a spoofed browser string.
# Measured 2026-07-28: a Chrome UA against DuckDuckGo returns 202 + a "select
# all squares containing a duck" CAPTCHA on every request, while this one still
# gets 200s. Spoofing makes us look like the scrapers they block. Wikipedia
# also requires a descriptive UA and 403s generic ones.
_SEARCH_UA = "Chitragupt/1.0 (hands-free cooking assistant; contact via repo)"

# Per-provider ceiling, not a total. The chain below tries providers in turn,
# so this is what one dead provider costs before we move on. 10s used to be the
# whole budget and a cold TLS handshake alone measured 6.4s.
_SEARCH_TIMEOUT = 8.0

MAX_SEARCH_RESULTS = 5


class SearchBlocked(Exception):
    """A provider refused to answer — CAPTCHA, rate limit, transport error.

    Distinct from "the provider answered and there genuinely is nothing", and
    the distinction is the entire point of this class. The old implementation
    collapsed both into `No web search results found for "X"`, because DDG
    serves its CAPTCHA page with HTTP *202* — a 2xx, so `raise_for_status()`
    stayed quiet, the `.result` selector matched nothing, and a hard block was
    reported to the model as an authoritative absence. On "does this contain
    beef?" that is the exact false-negative the anti-false-absence rule in the
    reasoning prompt exists to prevent, arriving via a tool result the model
    has no reason to doubt. Blocked must never be phrased as empty.
    """


def _ddg_is_challenge(resp) -> bool:
    """DDG signals its bot challenge with 202 + a duck-CAPTCHA body."""
    return resp.status_code == 202 or "bots use DuckDuckGo" in resp.text


def _is_excluded(url: str) -> bool:
    """Whether `url` is from a domain the operator has ruled out as a source.

    Providers below all return (title, url, snippet) triples specifically so
    this can be applied at one choke point in tool_web_search rather than
    four times with four chances to differ.

    Suffix-matched against the host, so "wikipedia.org" also covers
    en.wikipedia.org and de.m.wikipedia.org, while refusing to match a
    lookalike host like "notwikipedia.org.example.com".
    """
    from urllib.parse import urlparse
    from ..config import settings

    if not settings.SEARCH_EXCLUDED_DOMAINS:
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in settings.SEARCH_EXCLUDED_DOMAINS)


def _search_brave(query: str, client) -> list[str]:
    """Brave Search API — real index, 2,000 queries/month on the free tier.

    Only tried when BRAVE_API_KEY is set. It is the one provider here with a
    contract behind it; everything below is scraping and can break without
    notice, which on Render's datacenter IPs is a matter of when.
    """
    from ..config import settings

    resp = client.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": MAX_SEARCH_RESULTS},
        headers={"X-Subscription-Token": settings.BRAVE_API_KEY, "Accept": "application/json"},
    )
    if resp.status_code != 200:
        raise SearchBlocked(f"Brave returned {resp.status_code}")
    return [
        (r.get("title", "").strip(),
         r.get("url", ""),
         BeautifulSoup(r.get("description", ""), "html.parser").get_text(" ", strip=True))
        for r in resp.json().get("web", {}).get("results", [])
    ]


def _search_mojeek(query: str, client) -> list[str]:
    """Mojeek — independent crawler, no key, and it does not CAPTCHA bots.

    Primary keyless provider for that last reason alone: DDG's challenge rate
    from a datacenter IP makes it unusable as a base. Results carry direct
    hrefs, so unlike DDG there is no redirect wrapper to unpick.
    """
    resp = client.get("https://www.mojeek.com/search", params={"q": query})
    if resp.status_code != 200:
        raise SearchBlocked(f"Mojeek returned {resp.status_code}")

    results = []
    for li in BeautifulSoup(resp.text, "html.parser").select("ul.results-standard li"):
        title_el = li.select_one("a.title") or li.select_one("h2 a")
        if not title_el:
            continue
        snippet_el = li.select_one("p.s")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append((title_el.get_text(strip=True), title_el.get("href", ""), snippet))
    return results


def _search_ddg_lite(query: str, client) -> list[str]:
    """DuckDuckGo's lite endpoint — same index as the old html one, smaller page.

    Kept as a fallback rather than the primary because of the CAPTCHA above.
    Links are `//duckduckgo.com/l/?uddg=<url-encoded>` redirect wrappers; those
    get unwrapped so fetch_page receives a real URL it can actually retrieve.
    """
    from urllib.parse import parse_qs, unquote, urlparse

    resp = client.get("https://lite.duckduckgo.com/lite/", params={"q": query})
    if _ddg_is_challenge(resp):
        raise SearchBlocked("DuckDuckGo served a bot challenge")
    if resp.status_code != 200:
        raise SearchBlocked(f"DuckDuckGo returned {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select("a.result-link")
    snippets = soup.select("td.result-snippet")

    results = []
    for i, link in enumerate(links):
        href = link.get("href", "")
        wrapped = parse_qs(urlparse(href).query).get("uddg")
        url = unquote(wrapped[0]) if wrapped else href
        snippet = snippets[i].get_text(" ", strip=True) if i < len(snippets) else ""
        results.append((link.get_text(strip=True), url, snippet))
    return results


def _search_ddg_instant(query: str, client) -> list[str]:
    """DuckDuckGo's official Instant Answer API — an encyclopaedic abstract,
    not web results.

    Last in the chain because coverage is narrow: it answers "carrageenan" well
    and "how long to soak rajma" not at all. But it is a sanctioned JSON API
    rather than scraping, and it keeps returning usable content even while the
    scraped endpoints are serving challenges — which is exactly the situation
    where the chain has got this far.
    """
    resp = client.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1},
    )
    # This endpoint also stamps 202 on perfectly good JSON, so status is not a
    # usable block signal here — a parse failure is.
    try:
        data = resp.json()
    except ValueError:
        raise SearchBlocked("Instant Answer API returned non-JSON")

    results = []
    if data.get("AbstractText"):
        results.append((data.get("AbstractSource", "DuckDuckGo"),
                        data.get("AbstractURL", ""),
                        data["AbstractText"]))
    for topic in data.get("RelatedTopics", []):
        if topic.get("Text"):
            results.append((topic["Text"][:80], topic.get("FirstURL", ""), topic["Text"]))
    return results


def tool_web_search(query: str) -> str:
    """Search the web, trying providers in order until one actually answers.

    Chain rather than a single provider because every keyless option here is
    scraping someone who would rather we did not, and Render's egress is a
    datacenter IP that gets challenged far harder than a home connection. See
    SearchBlocked for why "blocked" and "no results" must stay distinguishable.
    """
    import httpx
    from ..config import settings

    providers: list[tuple[str, Any]] = []
    if settings.BRAVE_API_KEY:
        providers.append(("Brave", _search_brave))
    providers += [("Mojeek", _search_mojeek), ("DuckDuckGo", _search_ddg_lite),
                  ("DuckDuckGo Instant Answer", _search_ddg_instant)]

    failures = []
    with httpx.Client(
        headers={"User-Agent": _SEARCH_UA},
        timeout=_SEARCH_TIMEOUT,
        follow_redirects=True,
    ) as client:
        for name, search in providers:
            try:
                results = search(query, client)
            except (SearchBlocked, httpx.HTTPError, ValueError, KeyError) as e:
                failures.append(f"{name}: {type(e).__name__ if not isinstance(e, SearchBlocked) else e}")
                continue

            # Filter before truncating, not after — otherwise dropping one
            # excluded hit from a page of ten leaves four results when nine
            # were available.
            kept = [r for r in results if not _is_excluded(r[1])][:MAX_SEARCH_RESULTS]
            if kept:
                return f'Web search results for "{query}":\n' + "\n".join(
                    f"- {title} ({url})\n  {snippet}" for title, url, snippet in kept
                )

            # Nothing usable from this provider. Distinguish the two reasons:
            # "it had results but they were all excluded sources" is a config
            # consequence, not an absence, and must not end up phrased as one
            # (same trap as SearchBlocked — see §3.6).
            failures.append(f"{name}: {'all results excluded' if results else 'no results'}")

    if all(f.endswith("no results") for f in failures):
        return (
            f'No web search results found for "{query}". '
            f"This means the search engines returned nothing, which is weak evidence — "
            f"do not treat it as confirmation that something does not exist."
        )
    if all(f.endswith(("no results", "all results excluded")) for f in failures):
        return (
            f'No usable web search results for "{query}" — every result came from a '
            f"source this assistant is configured not to use. The search engines did "
            f"return content, so this is NOT evidence that nothing exists. Answer from "
            f"your own knowledge and say it is unverified."
        )
    return (
        f'Web search is unavailable right now (tried: {"; ".join(failures)}). '
        f'This is a tool failure, NOT a result about "{query}" — you learned nothing '
        f"about the query. Answer from your own knowledge and say it is unverified, "
        f"or ask the user to check. Never report this as \"nothing found\"."
    )


def tool_fetch_page(url: str) -> str:
    """Fetch a web page and return its visible text content."""
    import httpx

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    if _is_excluded(url):
        # Refuse before spending the round trip, and say why — an excluded
        # domain is a deliberate policy, not a page that failed to load, and
        # the model should look elsewhere rather than retry.
        return (
            f"Not fetched: {url} is from a source this assistant is configured not "
            f"to use. This is a policy choice, not a fact about the page. Find "
            f"another source rather than retrying this one."
        )

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _SEARCH_UA},
            timeout=_SEARCH_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        # Same rule as web_search: a fetch failure is a tool failure, not a
        # finding about the page. Say so, or the model reports "the page had
        # no information about X" when it never saw the page at all.
        return (
            f"Failed to fetch page ({e}). This is a tool failure — you learned "
            f"nothing about what the page says. Do not describe its contents."
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return text


def tool_calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval — only allow math
        import math
        allowed = {"abs": abs, "round": round, "int": int, "float": float, "min": min, "max": max, "sum": sum, "math": math}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_time(timezone: str = "") -> str:
    """Current time in `timezone`, or the user's configured local zone.

    Was a stub that took `timezone` and always returned UTC regardless. In one
    real session the model asked for Asia/Kolkata twice, got UTC both times,
    and then reasoned out loud about offsets from a number it had been told was
    something else — the same shape as the search bug in DECISIONS.md §3.6: the
    tool answered a different question than the one asked, without saying so.

    Hence the two rules below. An unknown zone says it fell back rather than
    quietly returning UTC, and every reply names the zone it actually used, so
    a wrong answer is visible rather than silent.

    Phrased for speech first. The reply may be read aloud by TTS, and an ISO
    timestamp spoken verbatim is unusable — the human phrasing leads and the
    precise form trails for arithmetic.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    from ..config import settings

    requested = (timezone or "").strip()
    name = requested or settings.DEFAULT_TIMEZONE
    fallback_note = ""

    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        # Unknown zone. Answer in the configured local zone rather than UTC —
        # it is the better guess for the person asking — but say so plainly.
        try:
            zone = ZoneInfo(settings.DEFAULT_TIMEZONE)
            fallback_note = (
                f" (NOTE: '{name}' is not a recognised timezone, so this is "
                f"{settings.DEFAULT_TIMEZONE} instead — do not present it as {name}.)"
            )
            name = settings.DEFAULT_TIMEZONE
        except (ZoneInfoNotFoundError, ValueError):
            # Misconfigured DEFAULT_TIMEZONE. UTC is the last resort and still
            # gets labelled, never passed off as local time.
            from datetime import timezone as _tz
            zone, name = _tz.utc, "UTC"
            fallback_note = (
                f" (NOTE: neither '{requested}' nor the configured default "
                f"'{settings.DEFAULT_TIMEZONE}' is a recognised timezone — "
                "this is UTC.)"
            )

    now = datetime.now(zone)
    # "16:09 on Wednesday 29 July 2026" — no leading zero on the day, since
    # %-d/%#d differ across platforms and this string may be spoken.
    spoken = f"{now:%H:%M} on {now:%A} {now.day} {now:%B %Y}"
    abbrev = now.tzname() or ""
    return (
        f"{spoken} — {name}"
        + (f" ({abbrev})" if abbrev and abbrev != name else "")
        + f". ISO: {now.isoformat(timespec='seconds')}."
        + fallback_note
    )


def tool_start_timer(label: str, duration_seconds: int, context: str = "") -> str:
    """Start a persisted background timer (no LLM cost while it runs)."""
    from . import timers
    timer_id = timers.start_timer(label, int(duration_seconds), context)
    minutes = int(duration_seconds) // 60
    seconds = int(duration_seconds) % 60
    duration_str = f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"
    return f"Timer '{label}' started for {duration_str} (id: {timer_id})."


def tool_cancel_timer(label: str) -> str:
    """Cancel a running timer so it never fires."""
    from . import timers
    return timers.cancel_timer(label)


def tool_update_task_list(title: str, items: list) -> str:
    """Replace the current task document (like Claude Code's TodoWrite)."""
    from . import tasklist
    document = tasklist.set_document(title, items)
    counts: dict[str, int] = {}
    for it in document["items"]:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items()) or "empty"
    return f"Task list '{title}' updated ({summary})."


def tool_log_observation(item: str, note: str, found: bool = False, alert: bool = False) -> str:
    """Silently record a short fact against a task-list item — the substitute
    for re-describing the whole scene every turn: write the fact once, read
    it back later instead of needing the original image again.

    `alert` and `found` are two different things, and conflating them was a
    real bug (see tasklist.add_observation):

      alert=True  "say this out loud this turn." Pure delivery, no state
                  change. Exists because tool-calling models routinely
                  return an empty `content` alongside a tool call, so a tick
                  that only called log_observation went silent even when the
                  note mattered. agent.py routes this through the normal
                  tool-result follow-up call instead of trusting the model
                  to also write visible text in the same completion.
      found=True  "the thing being searched for is now on screen." Ends a
                  "Find X" goal: completes the item and (via agent.py's
                  goal_complete) closes the camera. Implies alert.

    Anything that is merely worth mentioning is `alert`, not `found`."""
    from . import tasklist
    return tasklist.add_observation(item, note, found=found)


def tool_retract_observation(item: str, note_match: str, reopen: bool = False) -> str:
    """Undo a logged observation the user has corrected, and optionally re-open
    an item that was completed on the strength of it.

    The counterpart to log_observation, which is append-only. Without this a
    wrong fact — "found the toor dal" when it was black-eyed beans — stayed in
    [Task list] permanently and got repeated for the rest of the session even
    after the user objected twice. See tasklist.retract_observation."""
    from . import tasklist
    return tasklist.retract_observation(item, note_match, reopen=reopen)


def tool_request_camera() -> str:
    """Marker tool — never executed for its return value. Its presence in a
    response is intercepted specially in agent.py to ask the client for a
    fresh frame, since the server has no way to reach into the browser's
    camera itself."""
    return "CAMERA_REQUESTED"


def tool_request_live_search(target: str) -> str:
    """Registers the find-goal (so the live-frame silence/observation
    machinery has something to check frames against) and returns a marker.
    Actually starting the camera stream + polling is intercepted specially
    in agent.py/the client, same pattern as request_camera — the server
    can't turn on the browser's camera itself.
    """
    from . import tasklist
    tasklist.start_find_task(target)
    return "LIVE_SEARCH_REQUESTED"


def build_default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="Search the web for information",
        fn=tool_web_search,
        parameters={"query": {"type": "string", "description": "Search query", "required": True}},
        blocking=True,
    ))
    registry.register(Tool(
        name="fetch_page",
        description="Fetch a web page by URL and return its visible text content (e.g. to read the steps or details behind a search result)",
        fn=tool_fetch_page,
        parameters={"url": {"type": "string", "description": "The URL to fetch", "required": True}},
        blocking=True,
    ))
    registry.register(Tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        fn=tool_calculate,
        parameters={"expression": {"type": "string", "description": "Math expression", "required": True}},
    ))
    registry.register(Tool(
        name="get_time",
        description=(
            "Get the current date and time. Defaults to the user's local "
            "timezone, so omit the argument unless they explicitly ask about "
            "somewhere else — 'what time is it' means their time. The reply "
            "always names the zone it used; trust that over any offset you "
            "work out yourself."
        ),
        fn=tool_get_time,
        parameters={"timezone": {"type": "string", "description": "IANA timezone name such as 'Europe/Berlin' or 'Asia/Kolkata'. Leave empty for the user's local timezone.", "required": False}},
    ))
    registry.register(Tool(
        name="start_timer",
        description=(
            "Start a background timer for any step or wait period (e.g. 'boil eggs 10 min', "
            "'take the laundry out in 40 min', 'stand-up at 10:30'). "
            "Runs entirely server-side at no cost. When it completes, a follow-up message with the "
            "next step is generated automatically and delivered to the client — no need to check on it yourself."
        ),
        fn=tool_start_timer,
        parameters={
            "label": {"type": "string", "description": "Short name for what's being timed, e.g. 'eggs', 'laundry', 'call Sam'", "required": True},
            "duration_seconds": {"type": "integer", "description": "How many seconds to wait", "required": True},
            "context": {"type": "string", "description": "Relevant task context to reference when the timer completes", "required": False},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="cancel_timer",
        description=(
            "Cancel a running timer so it never goes off — for when the user says they don't "
            "want it, changed their mind, finished early, or is abandoning the step. The running "
            "timers are listed under [Timers] in your context; pass the label exactly as shown "
            "there. If nothing matches, or the label is ambiguous, you'll be told rather than "
            "having the wrong one cancelled — relay that and ask which they meant."
        ),
        fn=tool_cancel_timer,
        parameters={
            "label": {"type": "string", "description": "The timer's label exactly as shown in [Timers], or its id", "required": True},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="update_task_list",
        description=(
            "Create or update the current task document — the persistent record of what "
            "needs doing, what's done, and what's been substituted. Always send the FULL list of "
            "items every time, not just the one that changed (like rewriting a todo list in full "
            "on each edit) — items you omit are dropped. Statuses: 'pending' (not started), "
            "'in_progress' (currently doing), 'completed' (done — keep it in the list, don't remove "
            "it), 'skipped' (substituted or skipped — put why in 'note'). Use this instead of "
            "repeating the whole plan back to the user in every reply."
        ),
        fn=tool_update_task_list,
        parameters={
            "title": {"type": "string", "description": "Name of the overall task, e.g. 'Chicken Biryani', 'Replace laptop battery', 'Weekly shop'", "required": True},
            "items": {
                "type": "array",
                "description": "Full list of task items, sent in full every time — anything omitted is dropped.",
                "required": True,
                # Structural schema, not just prose — this is what actually
                # constrains native tool calling to the right field names
                # (was previously only described in free text, which let
                # the model drift to writing "task" instead of "content").
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The item's text — always use this exact key, never 'task' or 'label'"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "skipped"]},
                        "note": {"type": "string", "description": "Optional, e.g. reason for a substitution"},
                        "watch_for": {
                            "type": "string",
                            "description": (
                                "Optional. Your standing instruction to the camera for this step. It "
                                "is re-sent with every frame until you change it, so write it as the "
                                "question you want answered continuously. "
                                "You cannot see the camera yourself: a separate vision model looks on "
                                "your behalf, and it knows nothing about the task, the conversation, "
                                "or what 'correct' looks like — it only reports what is literally in "
                                "frame. So ask it for OBSERVATIONS, never for judgement. Ask 'how thick "
                                "are the slices', not 'are they being cut properly'; it reports, you "
                                "supply the critique and the advice from what it reports back. "
                                "Ask for everything the step needs in one brief — usually what is "
                                "happening, how far along it is, and how you will know it is done. "
                                "e.g. 'Describe how the onion is being cut: slice thickness, how even "
                                "the slices are, and roughly what fraction is still uncut. Say plainly "
                                "when the whole onion is chopped.' or 'Read the ingredient list on the "
                                "package the user is holding and list any of: beef, pork, lard, "
                                "gelatin, animal rennet.' "
                                "Set it on the item that is in_progress, and rewrite it when the step "
                                "changes."
                            ),
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["coarse", "fine"],
                            "description": (
                                "How closely the camera must look for this step. Default 'coarse' — "
                                "enough to tell what something is, where it is, and what is "
                                "happening. Use 'fine' ONLY when the answer depends on small print "
                                "or small differences the camera would otherwise miss: reading an "
                                "ingredient list, a label, a price, a model or serial number, a "
                                "measurement on a dial, or telling near-identical items apart. "
                                "'fine' sends the camera at full resolution and costs several times "
                                "more per frame, so set it back to 'coarse' (or complete the step) "
                                "once the close look is done — don't leave it on for a whole "
                                "session."
                            ),
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="log_observation",
        description=(
            "Silently record a short factual note against a task-list item — "
            "e.g. what you just saw relevant to it. Does not produce a reply "
            "to the user by itself; call this on every frame that's relevant "
            "to an active item, whether or not you also decide to say "
            "something out loud this turn. Use alert=true when the note is "
            "worth telling the user right now — that guarantees they hear it "
            "even if you write no visible text this turn. Use found=true ONLY "
            "to end a 'Find X' search because the target is now on screen; it "
            "completes that item and turns the camera off, so never use it for "
            "ordinary progress on a step."
        ),
        fn=tool_log_observation,
        parameters={
            "item": {"type": "string", "description": "The exact task-list item content (or its id) this observation is about", "required": True},
            "note": {"type": "string", "description": "Short factual note, e.g. 'freezer drawer open, chicken tenders visible, no ice cream'", "required": True},
            "alert": {"type": "boolean", "description": "True to say this note out loud to the user this turn. Use for anything noteworthy — progress, a problem, something they should know. Changes nothing in the task list.", "required": False},
            "found": {"type": "boolean", "description": "True ONLY if this is a 'Find X' item and the target is now visible. Marks that search complete and closes the camera. Never use it to report progress on a step — use alert instead.", "required": False},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="retract_observation",
        description=(
            "Undo something you previously logged that turns out to be wrong — "
            "ALWAYS call this when the user corrects a fact you recorded or "
            "told them (e.g. 'no, that's black eyed beans, not toor dal'). "
            "Removes the false note so you stop repeating it, and with "
            "reopen=true also moves an item you wrongly marked completed back "
            "to in progress. Correcting yourself in conversation is not enough: "
            "an uncorrected note keeps being fed back to you every turn and you "
            "will repeat it."
        ),
        fn=tool_retract_observation,
        parameters={
            "item": {"type": "string", "description": "The exact task-list item content (or its id) carrying the wrong note", "required": True},
            "note_match": {"type": "string", "description": "A distinctive fragment of the wrong note, e.g. 'toor dal'. Every observation containing it is removed.", "required": True},
            "reopen": {"type": "boolean", "description": "True if the item was marked completed/skipped because of this wrong note — moves it back to in_progress and clears a matching note.", "required": False},
        },
        needs_followup=False,
    ))
    registry.register(Tool(
        name="request_camera",
        description=(
            "Ask for a single fresh camera frame when answering the current "
            "message genuinely requires seeing the scene right now and no "
            "image is attached to this message. One frame, one look — use "
            "request_live_search instead if a single frame won't be enough. "
            "Only usable when no image was already provided this turn. Do "
            "not guess an answer that depends on the current scene without "
            "calling this first. This is one of several tools available to "
            "you, not a headline feature — don't volunteer it unprompted."
        ),
        fn=tool_request_camera,
        parameters={},
        needs_followup=False,
    ))
    registry.register(Tool(
        name="request_live_search",
        description=(
            "Start continuously watching the camera when the user needs "
            "help locating a specific physical object and a single frame "
            "won't be enough — they'll need to move the camera around "
            "while you keep checking. This ONLY watches for the named "
            "target; do not use it for general guidance, task "
            "tracking, or any other kind of ongoing help — that's not "
            "enabled through this tool. Once started, stay silent on "
            "frames that don't show the target (the live-frame protocol "
            "handles this automatically) and speak up only when you "
            "actually see it, or if the user seems to be searching the "
            "wrong place. This is one of several tools available to you, "
            "not a headline feature — don't volunteer it unprompted."
        ),
        fn=tool_request_live_search,
        parameters={
            "target": {"type": "string", "description": "The specific thing to look for, in the user's own words", "required": True},
        },
        needs_followup=False,
    ))
    return registry


# ─── Conversation Memory ──────────────────────────────────────────────────────

class ConversationMemory:
    """Simple in-memory conversation history."""

    def __init__(self, max_turns: int = 50):
        self.history: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self) -> list[dict]:
        return self.history

    def clear(self):
        self.history = []
