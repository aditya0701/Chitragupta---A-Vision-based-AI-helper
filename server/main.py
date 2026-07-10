"""Chitragupt — Vision-based Agentic Assistant API Server."""

from __future__ import annotations
import base64
import io
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# ─── Agent singleton ──────────────────────────────────────────────────────────

agent: Optional[ChitraguptAgent] = None


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

    agent = get_agent()
    try:
        result = await agent.process(
            image_base64=request.image_base64,
            prompt=request.prompt,
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

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTMLResponse(content=_WEB_UI, status_code=200)


_WEB_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chitragupt — Jarvis Vision Assistant</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid #2a2a4a;
  }
  header h1 {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  header .badge {
    font-size: 11px;
    background: #2a2a4a;
    padding: 2px 10px;
    border-radius: 12px;
    color: #888;
  }
  .main-container {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .sidebar {
    width: 280px;
    background: #0d0d1a;
    border-right: 1px solid #1a1a2e;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .sidebar h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555;
  }
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff8866;
  }
  .status-dot.offline { background: #ff4444; box-shadow: 0 0 8px #ff444466; }
  .tools-list { list-style: none; font-size: 13px; }
  .tools-list li { padding: 4px 0; color: #aaa; }
  .tools-list li::before { content: '⚡ '; }
  .chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    font-size: 14px;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .message.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #1a3a5c, #2a5a8c);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .message.assistant {
    align-self: flex-start;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-bottom-left-radius: 4px;
  }
  .message.assistant .model-tag {
    font-size: 11px;
    color: #666;
    margin-top: 6px;
  }
  .message.assistant .tool-msg {
    font-size: 12px;
    color: #ffaa00;
    margin-top: 4px;
  }
  .input-area {
    padding: 16px 24px;
    border-top: 1px solid #1a1a2e;
    background: #0d0d1a;
  }
  .input-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
  }
  .input-row textarea {
    flex: 1;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px 16px;
    color: #e0e0e0;
    font-size: 14px;
    resize: none;
    min-height: 48px;
    max-height: 120px;
    outline: none;
    font-family: inherit;
  }
  .input-row textarea:focus { border-color: #3a7bd5; }
  .input-row button {
    background: linear-gradient(135deg, #00d2ff, #3a7bd5);
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    color: #fff;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: transform 0.15s, opacity 0.15s;
  }
  .input-row button:hover { opacity: 0.9; transform: scale(1.02); }
  .input-row button:disabled { opacity: 0.4; cursor: not-allowed; }
  .image-preview {
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .image-preview img {
    height: 60px;
    border-radius: 8px;
    border: 1px solid #2a2a4a;
  }
  .image-preview .remove-img {
    background: none;
    border: none;
    color: #ff4444;
    cursor: pointer;
    font-size: 18px;
  }
  .upload-btn {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px;
    color: #aaa;
    cursor: pointer;
    font-size: 18px;
    transition: border-color 0.15s;
  }
  .upload-btn:hover { border-color: #3a7bd5; }
  #file-input { display: none; }
  .typing-indicator {
    align-self: flex-start;
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    border-bottom-left-radius: 4px;
  }
  .typing-indicator span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3a7bd5;
    animation: bounce 1.4s infinite;
  }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
  }
</style>
</head>
<body>
<header>
  <h1>✦ Chitragupt</h1>
  <span class="badge">Agentic Vision Assistant</span>
</header>
<div class="main-container">
  <div class="sidebar">
    <h3>Status</h3>
    <div class="status-indicator">
      <span class="status-dot" id="status-dot"></span>
      <span id="status-text">Connecting...</span>
    </div>
    <h3>Available Tools</h3>
    <ul class="tools-list">
      <li>web_search</li>
      <li>calculate</li>
      <li>get_time</li>
    </ul>
    <h3>Commands</h3>
    <button onclick="resetConversation()" style="background:#2a2a4a;border:none;border-radius:8px;padding:8px;color:#e0e0e0;cursor:pointer;font-size:13px;">↻ Reset Conversation</button>
  </div>
  <div class="chat-area">
    <div class="messages" id="messages">
      <div class="message assistant">
        Hello! I'm Chitragupt, your vision-enabled agentic assistant. Upload an image and ask me anything, or just chat!
        <div class="model-tag">Chitragupt v1.0</div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-row">
        <button class="upload-btn" onclick="document.getElementById('file-input').click()" title="Upload image">📷</button>
        <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
        <textarea id="prompt-input" rows="1" placeholder="Ask me anything..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
        <button id="send-btn" onclick="sendMessage()">Send →</button>
      </div>
      <div class="image-preview" id="image-preview" style="display:none;">
        <img id="preview-img" src="" alt="Preview">
        <button class="remove-img" onclick="clearImage()">✕</button>
      </div>
    </div>
  </div>
</div>
<script>
let currentImageBase64 = null;
let isProcessing = false;

async function checkHealth() {
  try {
    const resp = await fetch('/health');
    if (resp.ok) {
      document.getElementById('status-dot').className = 'status-dot';
      document.getElementById('status-text').textContent = 'Connected';
    }
  } catch { /* ignore */ }
}
checkHealth();

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    currentImageBase64 = e.target.result.split(',')[1];
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('image-preview').style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentImageBase64 = null;
  document.getElementById('image-preview').style.display = 'none';
  document.getElementById('preview-img').src = '';
  document.getElementById('file-input').value = '';
}

function addMessage(role, content, extras) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message ' + role;
  let html = content.replace(/\n/g, '<br>');
  if (extras && extras.model) {
    html += '<div class="model-tag">' + extras.model + '</div>';
  }
  if (extras && extras.tool_calls && extras.tool_calls.length > 0) {
    extras.tool_calls.forEach(tc => {
      html += '<div class="tool-msg">⚡ Used tool: ' + tc.tool + '</div>';
    });
  }
  div.innerHTML = html;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function showTyping() {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'typing-indicator';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function sendMessage() {
  if (isProcessing) return;
  const input = document.getElementById('prompt-input');
  const prompt = input.value.trim();
  if (!prompt && !currentImageBase64) return;

  isProcessing = true;
  document.getElementById('send-btn').disabled = true;

    addMessage('user', prompt || '(image uploaded)', {});
  input.value = '';
  showTyping();

  try {
    const resp = await fetch('/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt || 'What do you see in this image?',
        image_base64: currentImageBase64,
      }),
    });
    hideTyping();
    const data = await resp.json();

    if (data.scene_unchanged) {
      addMessage('assistant', '👁️ Scene unchanged — skipping reasoning. Still watching...', {
        model: data.provider + '/' + data.model,
      });
    } else {
      let displayText = data.text || '...';
      if (data.think_blocks && data.think_blocks.length > 0) {
        displayText += '\n\n<details><summary>💭 Thinking</summary>\n' + data.think_blocks.join('\n') + '\n</details>';
      }
      addMessage('assistant', displayText, { model: data.provider + '/' + data.model, tool_calls: data.tool_calls });
    }

    clearImage();
  } catch (err) {
    hideTyping();
    addMessage('assistant', '⚠️ Error: ' + err.message, {});
  }

  isProcessing = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}

async function resetConversation() {
  await fetch('/v1/reset', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
  addMessage('assistant', 'Conversation reset. How can I help you?', {});
}
</script>
</body>
</html>"""
