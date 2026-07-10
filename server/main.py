"""Chitragupt — Vision-based Agentic Assistant API Server."""

from __future__ import annotations
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .backends.factory import get_backend
from .agent.agent import ChitraguptAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chitragupt")

app = FastAPI(
    title="Chitragupt API",
    description="Vision-based Agentic Assistant — like Jarvis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─── Agent singleton ──────────────────────────────────────────────────────────

agent: Optional[ChitraguptAgent] = None

# Minimum seconds between accepted live-frame requests — a safety net behind
# the client-side interval/diff gate, in case a client misbehaves and hammers
# the endpoint (protects the Gemini free-tier quota).
LIVE_FRAME_MIN_INTERVAL_S = 1.5
_last_live_frame_time: float = 0.0


def get_agent() -> ChitraguptAgent:
    global agent
    if agent is None:
        backend = get_backend()
        agent = ChitraguptAgent(backend=backend)
        logger.info(f"Initialized agent with backend: {settings.BACKEND_MODE}")
    return agent


# ─── Request/Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None
    is_live_frame: bool = False


class ChatResponse(BaseModel):
    text: Optional[str] = None
    model: str
    provider: str
    tool_calls: list = []
    scene_unchanged: bool = False
    scene_description: Optional[str] = None
    think_blocks: list[str] = []


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mode": settings.BACKEND_MODE}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the agent. Optionally include a base64-encoded image."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    global _last_live_frame_time
    if request.is_live_frame:
        now = time.monotonic()
        if now - _last_live_frame_time < LIVE_FRAME_MIN_INTERVAL_S:
            return ChatResponse(
                text=None,
                model="n/a",
                provider="n/a",
                scene_unchanged=True,
            )
        _last_live_frame_time = now

    agent = get_agent()
    try:
        result = await agent.process(
            image_base64=request.image_base64,
            prompt=request.prompt,
            is_live_frame=request.is_live_frame,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return ChatResponse(
            text=f"Error: {e}",
            model="unknown",
            provider="error",
        )


@app.post("/v1/chat/upload")
async def chat_with_upload(
    prompt: str = Form(...),
    file: UploadFile = File(None),
):
    """Chat with the agent, uploading an image file directly."""
    image_base64 = None
    if file and file.content_type and file.content_type.startswith("image/"):
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

    agent = get_agent()
    result = await agent.process(
        image_base64=image_base64,
        prompt=prompt,
    )
    return ChatResponse(**result)


@app.post("/v1/reset")
async def reset_conversation():
    """Reset the agent's conversation memory."""
    agent = get_agent()
    agent.reset_conversation()
    return {"status": "conversation reset"}


# ─── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/sw.js")
async def service_worker():
    """Served from root scope so it can control the whole app."""
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/")
async def web_ui():
    return FileResponse(STATIC_DIR / "index.html")
