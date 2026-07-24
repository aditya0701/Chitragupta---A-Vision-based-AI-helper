"""Live tick-based world-doc system — the parallel workflow to server/agent.

Design (2026-07-16): the world document is the primary state and speech is a
side-effect of it. Ticks (camera frames on an interval) continuously update
the doc; a zero-LLM-cost trigger engine (expectation deadlines, staleness
arithmetic) decides when to wake the reasoning model; the model reads the doc
and decides whether anything warrants speaking, asking, or acting.

Runs alongside the original prompt-driven agent (server/agent) — separate
routes (/v2/*), separate UI page (/live), separate persisted state
(server/data/live/), shared backends and Tool/ToolRegistry classes. Nothing
here is imported by the old system.
"""
