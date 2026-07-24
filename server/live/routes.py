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
        _agent = LiveAgent(backend=backend)
        logger.info(f"Initialized live agent with backend mode: {mode or 'same as v1'}")
    return _agent


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
