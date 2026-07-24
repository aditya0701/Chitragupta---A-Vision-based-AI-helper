# ✦ Chitragupt — Agentic Vision Assistant

> *"Like Jarvis, but it can see."*

**Chitragupt** is a vision-enabled agentic AI assistant that can see images, reason about them, use tools, and maintain conversation context. It runs a Vision Language Model (VLM) on **Google Colab's GPU** (free tier!) and exposes it via a local API server with a beautiful web UI, desktop app, and CLI.

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   🌐 Web UI     │     │   🖥️ Desktop App  │     │   💻 CLI        │
│  (built-in)     │     │  (Tkinter GUI)    │     │  (terminal)     │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         └───────────────┬───────┴────────────┬────────────┘
                         │                    │
                  ┌──────▼──────┐      ┌──────▼──────┐
                  │  FastAPI    │      │  FastAPI     │
                  │  Local      │◄────►│  Colab       │
                  │  Server     │      │  (ngrok)     │
                  │  :8000      │      │              │
                  └──────┬──────┘      └──────┬──────┘
                         │                    │
                  ┌──────▼──────┐      ┌──────▼──────┐
                  │  Backends   │      │  VLM Model   │
                  │  (factory)  │      │  (LLaVA,     │
                  │             │      │   Qwen-VL,   │
                  │  ┌──────┐   │      │   etc.)      │
                  │  │Colab │───┼──────►             │
                  │  ├──────┤   │      └─────────────┘
                  │  │OpenAI│   │
                  │  ├──────┤   │
                  │  │Anthrop│  │
                  │  ├──────┤   │
                  │  │Ollama │  │
                  │  └──────┘   │
                  └─────────────┘
```

### Modes

| Mode | Description | Cost |
|------|-------------|------|
| **Colab** (default) | VLM runs on Google Colab GPU, exposed via ngrok | Free (T4 GPU) |
| **API** | Uses cloud APIs (OpenAI GPT-4o, Anthropic Claude, Gemini) | API costs apply |
| **Local** | Uses Ollama with a local vision model | Free (your GPU) |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repo
cd AI_Chitragupt

# Set up the server
cd server
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings
```

### 2. Choose Your Backend

#### Option A: Google Colab (Free GPU) ⭐ Recommended

1. Open [`colab/chitragupt_colab_vlm.ipynb`](colab/chitragupt_colab_vlm.ipynb) in Google Colab
2. Add your `NGROK_AUTH_TOKEN` to Colab secrets (get one free at [ngrok.com](https://ngrok.com))
3. Run all cells — it will load a VLM and give you a public URL
4. Copy the ngrok URL into your `.env`:

```env
BACKEND_MODE=colab
COLAB_API_URL=https://your-ngrok-url.ngrok.io
COLAB_API_KEY=chitragupt-secret-key
```

#### Option B: OpenAI / Anthropic / Gemini

```env
BACKEND_MODE=api
API_PROVIDER=openai
API_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

#### Option C: Local Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull llava:13b
```

```env
BACKEND_MODE=local
OLLAMA_MODEL=llava:13b
```

### 3. Start the Server

```bash
cd server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or simply:

```bash
cd server
python -m server.main
```

### 4. Open the Web UI

Visit **[http://localhost:8000](http://localhost:8000)** — a beautiful dark-themed chat interface.

---

## 🎮 Usage

### Web UI

The built-in web UI at `http://localhost:8000` provides:
- 💬 Chat interface with markdown-style messages
- 📷 Image upload with preview
- ⚡ Real-time tool usage indicators
- 🎨 Dark theme with cyberpunk aesthetic
- 🔄 Conversation reset

### Desktop App

```bash
cd client
pip install -r requirements.txt
python desktop.py
```

### CLI

```bash
# One-shot
python client/cli.py "What's in this image?" --image photo.jpg

# Interactive mode
python client/cli.py --interactive

# With custom server URL
python client/cli.py --url http://localhost:8000 --interactive
```

### API

```bash
# Health check
curl http://localhost:8000/health

# Chat with image (base64)
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What do you see?",
    "image_base64": "<base64-encoded-image>"
  }'

# Chat with file upload
curl -X POST http://localhost:8000/v1/chat/upload \
  -F "prompt=Describe this image" \
  -F "file=@photo.jpg"

# Reset conversation
curl -X POST http://localhost:8000/v1/reset
```

---

## 🧠 Agentic Capabilities

Chitragupt has built-in tools that the VLM can invoke:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for information |
| `calculate` | Evaluate mathematical expressions |
| `get_time` | Get the current time |

The model uses a special ````tool` block to request tool execution:

````
What's 42 * 13?

```tool
{"name": "calculate", "arguments": {"expression": "42 * 13"}}
```
````

---

## 🗂️ Project Structure

```
AI_Chitragupt/
├── server/                    # Local API server
│   ├── main.py               # FastAPI app + Web UI
│   ├── config.py             # Settings loader
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # Environment config template
│   ├── backends/
│   │   ├── __init__.py       # Abstract base class
│   │   ├── factory.py        # Backend selector
│   │   ├── colab.py          # Colab VLM backend
│   │   ├── openai_backend.py # OpenAI backend
│   │   ├── anthropic_backend.py # Anthropic backend
│   │   ├── gemini_backend.py # Gemini backend
│   │   └── ollama_backend.py # Ollama backend
│   └── agent/
│       ├── __init__.py       # Tool registry, memory
│       └── agent.py          # Main agent loop
├── colab/
│   └── chitragupt_colab_vlm.ipynb  # Colab notebook
├── client/
│   ├── cli.py                # Terminal client
│   ├── desktop.py            # Tkinter desktop app
│   └── requirements.txt
└── README.md
```

---

## 🛠️ Extending

### Add a new tool

Edit `server/agent/__init__.py`:

```python
def tool_my_function(arg: str) -> str:
    """Do something useful."""
    return f"Result: {arg}"

registry.register(Tool(
    name="my_function",
    description="Describe what it does",
    fn=tool_my_function,
    parameters={"arg": {"type": "string", "description": "...", "required": True}},
))
```

### Add a new backend

1. Create `server/backends/my_backend.py` implementing `VisionBackend`
2. Add it to `server/backends/factory.py`

---

## 📋 Requirements

### Server
- Python 3.10+
- FastAPI, Uvicorn, httpx

### Colab
- Google account (free)
- ngrok account (free)

### Client
- Python 3.10+
- Tkinter (included with Python on most systems)

---

## 🧾 Recent Changes & Roadmap

> This section is the running "what changed, what's next, and **why**" log for
> the live camera-assistant work. It's driven by real cooking sessions (the
> primary use case: hands-free kitchen help over voice/text), so most entries
> cite the actual issue that motivated the change rather than a feature idea in
> the abstract. For the deep implementation detail, see `CLAUDE.md`.

### ✅ Changed — 2026-07-24 (from a full dal-makhani cooking session)

Real end-to-end use went well overall — the assistant helped find and soak the
dal, put the pressure cooker on, and add eggs + sausages, and the camera
open/close behaviour worked cleanly. That session also surfaced a batch of
usability bugs, now fixed:

- **Speak-after-task.** *Issue:* the model would set up a task list, show a
  "⚡ Used tool" blob, but say nothing back — leaving the user waiting with no
  spoken reply, unable to move forward without re-prompting. *Fix:* when a turn
  only calls a side-effect tool (`update_task_list` / `start_timer`) and
  produces no visible text, the server now makes one follow-up call (feeding the
  tool result back) so the model actually tells you what it set up and what to
  do next — "like a fresh tick." It only fires when the model went silent, so
  normal turns cost nothing extra, and live watching stays silent as before.

- **Timer no longer starts preemptively.** *Issue:* the user said "I want to
  boil eggs" and the assistant immediately started a timer — before the eggs
  were anywhere near the stove. *Fix:* timers now only start once a step has
  genuinely begun (you say so, or it's visible on camera); a planning statement
  gets an acknowledgement and a request for confirmation instead.

- **Task steps no longer fed to the vision model as things to "find".** *Issue:*
  for a "Find dal" goal, the reasoning model was handing the vision model the
  whole task step ("Soak urad and rajma dal, then…") as the detection target,
  which was slow and unfocused. *Fix:* only genuine "Find X" goals drive object
  detection now, with the "Find " prefix stripped to the bare object; ordinary
  cooking steps get a short, step-relevant scene description instead. Also made
  the vision prompt an explicit object-detection directive, which measurably sped
  up the vision stage.

- **Crash-safe conversation cache.** *Issue:* the user accidentally hit browser
  "back" mid-session and lost the visible conversation. *Fix:* the chat is now
  mirrored to `localStorage` and restored on reload/close/back ("↑ restored from
  your last session"). The task list already survived (it's persisted
  server-side), so between the two your cooking progress is protected. Applied to
  both the normal UI and the debug UI (the debug UI persists a slimmed copy — it
  deliberately doesn't cache the heavy raw-pipeline dumps).

- **Debug `.md` export now includes the full pipeline/wire log.** *Issue:* the
  exported transcript dropped "the post and other stuff" — the POST requests,
  the vision-stage prompt/description, the tool calls — making it hard to debug
  from an exported report. *Fix:* every wire-log line (POST, `[vision→Qwen]` /
  `[vision←Qwen]`, tool start/result, done event) is captured and dumped as a
  time-ordered `## Pipeline / wire log` block in the export.

- **Earlier the same session:** unified the UI into one screen (removed the
  separate Live Watch mode — the camera is simply on or off, and turning it on
  starts hands-free watching in place), fixed the silence protocol leaking prose
  narration ("staying silent…") into the chat, fixed the camera not auto-closing
  once a find-goal completed, and added a read-only task-list panel so
  model-created tasks (which always drove the prompt but had no UI) are visible.

### 🔜 Will change — planned next

- **V2: camera-driven proactive timers.** Instead of waiting to be told, the
  assistant should notice from the camera that a step has actually started ("I
  can see the eggs are on — want me to set a timer?") and only then offer/start
  it. *Why:* it's the natural extension of the "don't start timers preemptively"
  fix above — the confirmation should be able to come from what it *sees*, not
  only from what you type. This is a bigger event-condition feature in the v2
  world-doc trigger system (`server/live/`), so it's its own pass.

- **Fix the streaming-blind vision path.** *Issue:* on the streaming chat
  endpoint, the DeepSeek reasoning call ignores the attached image — a typed
  question with a photo can be answered without the picture actually reaching the
  model. *Why it matters:* it's a correctness bug that can produce confident
  answers grounded in nothing.

- **TTS (spoken output).** Text-only by design until the text pipeline is fully
  trusted; planned via the browser's free Web Speech API. Pairs naturally with
  the speak-after-task fix — once the assistant reliably *says* something, having
  it read aloud is the hands-free payoff.

- **Faster / cheaper inference.** Investigating paid Qwen hosting and larger
  model versions vs. the current Groq free tier (capped at 8K tokens/minute,
  which is the main latency bottleneck today), and rented-GPU economics.

- **Longer-running reliability:** adaptive poll backoff and a keep-alive pinger
  so unattended timers survive Render's free-tier idle spin-down, and multi-timer
  progress in the UI.

### Why these changes at all

Almost every fix above traces to a specific failure in real hands-free cooking
use, not a spec. The throughline: when your hands are busy and you're relying on
voice/camera, the assistant has to (1) actually respond out loud when it does
something, (2) not act (timers) before the thing has really happened, (3) not
lose your progress to an accidental tap, and (4) be debuggable after the fact
from an exported log when something does go wrong. The roadmap continues in the
same direction — moving confirmation from "what you typed" toward "what it sees."

---

## 📜 License

MIT

---

## 🙏 Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM models from [Hugging Face](https://huggingface.co/)
- GPU acceleration by [Google Colab](https://colab.research.google.com/)
- Tunneling by [ngrok](https://ngrok.com/)
