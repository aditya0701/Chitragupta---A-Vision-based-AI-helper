# Vision Chitragupta

An egocentric live camera assistant with a two-stage vision + reasoning pipeline,
inspired by the Hindu mythological figure Chitragupta — the all-seeing record keeper
who observes, records, and reports.

The system watches through a camera, understands what it sees using a vision model,
reasons about it using a ReAct agent with chain-of-thought thinking, and responds
via text and optional TTS.

---

## Architecture

The pipeline is deliberately split into two sequential model calls:

```
Phone/camera  →  frame (JPEG, base64)
                     ↓
              [Vision Model]           Qwen3-VL 8B via Ollama
              Describes what it sees   ~5 sec
                     ↓
              [Memory Buffer]          Rolling JS state (last 5-10 frames)
              Context + change detect
                     ↓
              [ReAct Reasoning]        Qwen3 8B via Ollama, /think mode
              Thinks, optionally       ~15-30 sec (thinking on)
              calls tools, responds    ~5 sec (thinking off)
                     ↓
              [TTS]                    Browser Web Speech API / Kokoro / ElevenLabs
```

### Why this split

The vision model's only job is turning pixels into words. It is fast, cheap,
and runs on every sampled frame. All intelligence lives in the reasoning layer,
which only runs when something is worth responding to.

### Why ReAct over pure orchestration

Pure orchestration (a controller that routes between specialist models) requires
predicting every situation in advance and breaks on edge cases. ReAct lets the
reasoning model orchestrate itself: it reads the scene, decides mid-thought whether
to call a tool, and routes to the right response type without a separate controller.
The thinking chain IS the orchestration.

---

## Current stack

| Layer | Tool | Notes |
|---|---|---|
| Vision model | Qwen3-VL 8B | Via Ollama, OpenAI-compatible API |
| Reasoning | Qwen3 8B | /think mode for chain-of-thought |
| Inference server | Ollama | CORS enabled: `OLLAMA_ORIGINS="*"` |
| Infrastructure | Google Colab T4 + ngrok | Free tier, 16GB VRAM |
| Frontend | React (browser artifact) | Calls Ollama via configurable URL |
| TTS | Browser Web Speech API | Free, zero setup |
| Camera | Phone browser or webcam | MediaStream API |

---

## Infrastructure setup

The project currently runs on Colab with ngrok tunneling.
See `colab_setup.ipynb` for the full notebook.

Quick summary of the Colab flow:

```python
# 1. Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# 2. Start with CORS
env['OLLAMA_ORIGINS'] = '*'
env['OLLAMA_HOST'] = '0.0.0.0'
subprocess.Popen(['ollama', 'serve'], env=env)

# 3. Pull models (cached to Drive after first pull)
!ollama pull qwen3-vl:8b
!ollama pull qwen3:8b

# 4. Expose via ngrok
tunnel = ngrok.connect(11434)
# Paste tunnel URL into frontend Settings > Ollama URL
```

Model cache is stored in Google Drive at `/content/drive/MyDrive/ollama_models`
to avoid re-downloading on every session (models are ~5GB each).

To run locally when a GPU is available:
```bash
OLLAMA_ORIGINS="*" ollama serve
```

Minimum viable GPU: 12GB VRAM to hold both models simultaneously.
Recommended: RTX 3090 (24GB, ~$450 used). The Colab T4 (16GB) works fine.

---

## Models

### Vision: Qwen3-VL 8B
- Handles image input natively
- Task: describe the scene in 100-300 words
- VRAM: ~5.5GB at Q4
- Pull: `ollama pull qwen3-vl:8b`

### Reasoning: Qwen3 8B
- Thinking mode enabled via `/think` prefix in user message
- Produces `<think>...</think>` chain followed by visible response
- Supports tool calling (function calling) natively
- VRAM: ~4.6GB at Q4
- Pull: `ollama pull qwen3:8b`

### Upgrade paths
- Reasoning quality: swap to `qwen3:14b` (~8.3GB) or `qwen3:30b-a3b` (MoE, ~17GB, runs at 3B speed)
- Vision quality: swap to `qwen3-vl:32b` (~24GB, needs full 24GB GPU)
- Speed: use `qwen3.5:9b` for reasoning (~5GB, beats older 8B on most benchmarks)

---

## Key design decisions

### Thinking is adaptive, not always on
The reasoning model decides whether to think based on the prompt complexity.
Simple questions skip the think chain for speed (~5 sec total).
Complex multi-step questions use full thinking (~25 sec total).
This is controlled by whether `/think` is prepended to the user message,
which the frontend exposes as a toggle.

### Camera is not processed every frame
The memory buffer compares consecutive frame descriptions.
If the scene has not meaningfully changed, the reasoning layer does not run.
This prevents constant output and keeps GPU usage reasonable.

### Phone as camera
The phone browser opens the artifact URL, grants camera access,
and sends frames to the Ollama backend via the ngrok URL.
No native app required.

### No separate controller
Earlier designs had a dedicated controller LLM that routed between handlers.
This was removed in favour of ReAct because it required predicting all routing
cases upfront and added a full extra LLM call. The reasoning model handles
its own routing within the thinking chain.

---

## File structure

```
/
├── CLAUDE.md                    This file
├── frontend/
│   └── VisionChitragupta.jsx    React frontend (camera, pipeline UI, settings)
├── notebooks/
│   └── colab_setup.ipynb        Colab notebook (Ollama + ngrok setup)
├── prompts/
│   ├── vision_prompt.txt        System prompt for vision model
│   └── reasoning_prompt.txt     System prompt for reasoning model (ReAct + tools)
└── tools/
    └── tool_definitions.json    Tool schemas for Qwen3 function calling
```

---

## Prompts

### Vision model prompt
Keep it simple. The vision model's only job is accurate description.

```
Describe everything visible in this image in detail.
Include: objects, people, actions, text, colours, spatial layout,
and anything that might matter for helping someone understand this scene.
Be factual and specific. Do not offer advice or opinions.
```

### Reasoning model system prompt

```
You are Chitragupta, an all-seeing assistant with access to tools.
You receive a description of what a camera currently sees,
plus any question from the user.

Think step by step before responding.
If you need external information, call a tool inside your thinking.
Be concise, practical, and helpful in your final response.

Available tools:
- search(query): web search for identifying unknown objects or facts
- calculate(expression): arithmetic and unit conversion
- translate(text, target_language): translate text visible in the image

To call a tool, write inside your think block:
<tool>search: red mushroom white spots</tool>
The result will be returned to you automatically.
```

---

## TODO

The following components are not yet built:

- [ ] ReAct prompt engine with `<tool>` tag parser
- [ ] Tool executor (search via SearXNG or Brave API, calculator, translate)
- [ ] Memory buffer with change detection between frames
- [ ] Streaming output (show thinking tokens as they arrive)
- [ ] Kokoro TTS integration (local, higher quality than Web Speech API)
- [ ] Egocentric fine-tuning pipeline (Ego4D + Egocentric-1M data)
- [ ] Local GPU deployment (port from Colab when hardware available)

---

## Fine-tuning (future)

Standard vision models were trained on third-person internet images.
Egocentric (first-person) footage from a phone camera has a different distribution:
hands in frame, close objects, shaky movement, indoor lighting.
Fine-tuning on egocentric data improves performance meaningfully.

Relevant datasets:
- Ego4D (Meta): 3670 hours, requires license agreement at ego4d.dev
- Egocentric-1M (Build AI): ~1M hours, Apache 2.0, released April 2026
- EPIC-Kitchens: 100 hours, kitchen-focused, densely annotated

Strategy: do not fine-tune from scratch. Identify failure cases on real camera
footage first, then fine-tune only on those failure patterns using targeted clips
from the datasets above.

---

## Non-Chinese model alternatives

The default stack uses Qwen (Alibaba, Chinese). If data residency or supply chain
concerns require non-Chinese models:

| Layer | Alternative | Notes |
|---|---|---|
| Vision | Gemma4 4B (Google) | Apache 2.0, runs on 16GB, good vision |
| Vision | Llama 4 Scout (Meta) | Image only, no video |
| Reasoning | Gemma4 27B (Google) | Apache 2.0, strong reasoning |
| Reasoning | Phi-4 (Microsoft) | MIT, 14B, strong reasoning for size |
| Full stack API | Gemini 2.0 Flash | Handles real video streams natively |

---

## Latency reference

| Mode | Vision call | Reasoning call | Total |
|---|---|---|---|
| Thinking ON (complex) | ~5 sec | ~20-30 sec | ~25-35 sec |
| Thinking OFF (simple) | ~5 sec | ~5 sec | ~10 sec |
| Adaptive (default) | ~5 sec | depends | ~10-35 sec |

Measured on Colab T4 with Qwen3-VL 8B + Qwen3 8B at Q4 quantization.
Speeds roughly double on RTX 3090.