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

## 📜 License

MIT

---

## 🙏 Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM models from [Hugging Face](https://huggingface.co/)
- GPU acceleration by [Google Colab](https://colab.research.google.com/)
- Tunneling by [ngrok](https://ngrok.com/)
