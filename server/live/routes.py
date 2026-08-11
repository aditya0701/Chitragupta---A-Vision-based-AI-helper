"""Routes for the live system — everything under /v2, plus the /live page.

Wired into the app by two include_router lines in server/main.py (the only
'connector' the old system needed). Both systems run simultaneously; they
share nothing but the backend classes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..backends.factory import get_backend
from . import config, worlddoc
from .agent import LiveAgent

logger = logging.getLogger("chitragupt.live")

router = APIRouter(prefix="/v2")
page_router = APIRouter()

STATIC_DIR = Path(__file__).parent.parent / "static"

_agent: Optional[LiveAgent] = None
_last_tick_time: float = 0.0


def get_live_agent() -> LiveAgent:
    global _agent
    if _agent is None:
        mode = None if config.LIVE_BACKEND_MODE == "same" else config.LIVE_BACKEND_MODE
        backend = get_backend(mode)
        _check_vision_provider(backend, mode)
        _agent = LiveAgent(backend=backend)
        # ASCII only. This is the one line an operator is told to check before
        # a session, and a Windows console renders an em dash as garbage.
        logger.info(
            "Initialized live agent | backend mode: %s | VISION ON: %s | reasoning: %s",
            mode or "same as v1", backend.VISION_PROVIDER,
            getattr(backend, "model", "?"),
        )
    return _agent


def _check_vision_provider(backend, mode: str | None) -> None:
    """Refuse to start v2 on a backend that sends frames to Groq.

    v2 cannot run there — one tick per 11s against the TPM cap versus a ~4s
    interval, and ~139 ticks TOTAL per day against the TPD cap. It is not slow
    or degraded, it is a session that ends in a 429 partway through.

    This is a startup check rather than a comment because the failure was
    invisible until it was expensive. On 2026-08-10 a live cooking session ran
    eighteen minutes and died mid-answer on Groq's daily cap, while `.env`,
    `render.yaml` and every doc in the repo said `deepinfra`. Nothing reads
    wrong in that picture — a stale process simply never picked the setting up,
    and there was no point at which anything compared intent against reality.

    Failing here costs one confusing startup and is immediately actionable.
    Failing eighteen minutes into someone's cooking costs the session, and
    they only find out because the assistant stops answering.
    """
    if backend.VISION_PROVIDER != "groq":
        return
    # Everything below is ASCII on purpose. This message is read off a console,
    # and a Windows cp1252 terminal raises UnicodeEncodeError on an em dash or a
    # section sign — so a diagnostic containing them dies while reporting the
    # problem it was written to explain.
    if config.ALLOW_GROQ_VISION:
        logger.warning(
            "v2 is running vision on GROQ by explicit LIVE_ALLOW_GROQ_VISION. "
            "Expect 429s within the hour - the free tier allows ~139 ticks per day."
        )
        return
    raise RuntimeError(
        f"v2 refuses to start with vision on Groq (backend mode {mode!r} -> "
        f"{type(backend).__name__}). The tick loop exhausts Groq's free tier in "
        f"under 20 minutes, so this configuration cannot work - see "
        f"DECISIONS.md section 5.2.\n"
        f"Fix: set LIVE_BACKEND_MODE=deepinfra (plus DEEPINFRA_API_KEY) in "
        f"server/.env, and RESTART the server - a process started before that "
        f"setting existed will not have picked it up, which is exactly how this "
        f"happened last time.\n"
        f"To override deliberately: LIVE_ALLOW_GROQ_VISION=true."
    )


class TickRequest(BaseModel):
    image_base64: str


class LiveChatRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None


@router.post("/tick")
async def tick(request: TickRequest):
    global _last_tick_time
    now = time.monotonic()
    if now - _last_tick_time < config.TICK_MIN_INTERVAL_S:
        return {"skipped": True, "text": None}
    _last_tick_time = now
    try:
        return await get_live_agent().tick(request.image_base64)
    except Exception as e:
        logger.error(f"Live tick error: {e}", exc_info=True)
        return {"text": None, "error": str(e)}


@router.post("/chat")
async def chat(request: LiveChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        return await get_live_agent().chat(request.prompt, request.image_base64)
    except Exception as e:
        logger.error(f"Live chat error: {e}", exc_info=True)
        return {"text": f"Error: {e}", "error": str(e)}


@router.get("/poll")
async def poll():
    """Trigger heartbeat — pure arithmetic unless an expectation just fired
    (one reasoning call then). Safe to call frequently; also what keeps the
    Render dyno awake during quiet stretches."""
    try:
        return await get_live_agent().poll()
    except Exception as e:
        logger.error(f"Live poll error: {e}", exc_info=True)
        return {"message": None, "error": str(e)}


@router.get("/doc")
async def doc():
    d = worlddoc.load()
    return {"rendered": worlddoc.render(d), "raw": d}


@router.post("/reset")
async def reset():
    get_live_agent().reset()
    return {"status": "live system reset"}


@page_router.get("/live")
async def live_ui():
    return FileResponse(STATIC_DIR / "live.html")
