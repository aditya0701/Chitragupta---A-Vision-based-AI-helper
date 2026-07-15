// Debug UI — same pipeline as the normal app, but every turn dumps exactly
// what was sent to the model and what came back, inline in the chat itself.
// No sidebar: settings live in a thin bar under the mode switcher, and
// everything worth *analyzing* (prompts, raw responses, tool calls, activity)
// is a block in the message stream, not tucked away somewhere you have to
// go looking for it.

let currentImageBase64 = null;
let isProcessing = false;

// ─── Raw-dump rendering — the actual point of this page ───────────────────

// Replaces base64 image payloads with a length note (unreadable as text, and
// dumping ~100KB of it per turn buries everything else) and caps any other
// very long string so one runaway tool result doesn't swallow the page.
function safeStringify(obj) {
  return JSON.stringify(obj, (key, value) => {
    if (key === 'image_base64' && typeof value === 'string') {
      return `<base64 image, ${value.length} chars, omitted>`;
    }
    if (typeof value === 'string' && value.length > 4000) {
      return value.slice(0, 4000) + `...[${value.length - 4000} more chars truncated]`;
    }
    return value;
  }, 2);
}

function makeSubBlock(title, obj) {
  const wrap = document.createElement('div');
  wrap.className = 'dbg-step';
  const label = document.createElement('div');
  label.className = 'dbg-step-label';
  label.textContent = title;
  const pre = document.createElement('pre');
  pre.textContent = safeStringify(obj);
  wrap.appendChild(label);
  wrap.appendChild(pre);
  return wrap;
}

// One block per actual backend.chat()/chat_stream() call the turn made
// (initial reasoning call, any truncation/rate-limit retry, the tool-result
// follow-up call, ...) — see agent.py's _record_debug_step. This is "what
// the model actually saw," not a summary of it.
function makeStepBlock(i, step) {
  const wrap = document.createElement('div');
  wrap.className = 'dbg-step';
  const label = document.createElement('div');
  label.className = 'dbg-step-label';
  label.textContent =
    `● call ${i + 1}: ${step.label}  (has_image=${step.has_image} think=${step.think}` +
    (step.tools_offered && step.tools_offered.length ? ` tools_offered=[${step.tools_offered.join(',')}]` : '') +
    ')';
  wrap.appendChild(label);

  const promptPre = document.createElement('pre');
  promptPre.textContent = 'PROMPT SENT:\n' + step.prompt_sent;
  wrap.appendChild(promptPre);

  const respPre = document.createElement('pre');
  let respText =
    `RESPONSE (${step.provider}/${step.model}${step.truncated ? ', TRUNCATED' : ''}):\n`;
  if (step.response_reasoning) respText += '[reasoning]\n' + step.response_reasoning + '\n\n';
  respText += '[visible text]\n' + (step.response_text || '(empty)');
  if (step.tool_calls_raw && step.tool_calls_raw.length) {
    respText += '\n\n[raw tool_calls from API]\n' + safeStringify(step.tool_calls_raw);
  }
  respPre.textContent = respText;
  wrap.appendChild(respPre);

  return wrap;
}

// The full raw picture for one turn: the request body, every model call
// made while handling it, every tool actually executed (with its full
// arguments and result — this is where a web_search query and its results
// show up, verbatim), and the final response envelope. Expanded by default
// on purpose — nothing here is worth hiding behind an extra click when the
// whole point of this page is to see it.
function renderDebugDump(container, requestObj, data) {
  const details = document.createElement('details');
  details.className = 'dbg-raw';
  details.open = true;
  const summary = document.createElement('summary');
  summary.textContent = '🔍 raw pipeline data';
  details.appendChild(summary);

  const body = document.createElement('div');
  body.appendChild(makeSubBlock('→ REQUEST', requestObj));

  const steps = (data.debug && data.debug.steps) || [];
  steps.forEach((s, i) => body.appendChild(makeStepBlock(i, s)));

  if (data.tool_calls && data.tool_calls.length) {
    body.appendChild(makeSubBlock('⚡ TOOL CALLS (executed, with results)', data.tool_calls));
  }

  if (data.vision_prompt || data.scene_description) {
    body.appendChild(makeSubBlock('👁 VISION STAGE (split backend only)', {
      vision_prompt: data.vision_prompt, scene_description: data.scene_description,
    }));
  }

  body.appendChild(makeSubBlock('← RESPONSE SUMMARY', {
    text: data.text, model: data.model, provider: data.provider,
    scene_unchanged: !!data.scene_unchanged,
    needs_camera: !!data.needs_camera,
    needs_live_search: !!data.needs_live_search,
    search_target: data.search_target || null,
    rate_limited: !!data.rate_limited,
    retry_after: data.retry_after || null,
  }));

  if (data.debug && data.debug.timer_completions && data.debug.timer_completions.length) {
    body.appendChild(makeSubBlock('⏰ TIMER COMPLETIONS FOLDED INTO THIS TURN', data.debug.timer_completions));
  }
  if (data.debug && data.debug.note) {
    body.appendChild(makeSubBlock('NOTE', data.debug.note));
  }

  details.appendChild(body);
  container.appendChild(details);
}

// ─── Conversation export ────────────────────────────────────────────────────
let transcriptLog = [];

function logTranscript(role, text, extras) {
  transcriptLog.push({
    role,
    text,
    model: extras && extras.model,
    tool_calls: (extras && extras.tool_calls) || [],
    think_blocks: (extras && extras.think_blocks) || [],
    debug: (extras && extras.rawData && extras.rawData.debug) || null,
    at: new Date().toISOString(),
  });
}

function exportConversation() {
  const lines = ['# Chitragupt debug transcript export', `Exported ${new Date().toISOString()}`, ''];
  transcriptLog.forEach((entry) => {
    lines.push(`## ${entry.role} (${entry.at})`);
    lines.push(entry.text);
    if (entry.model) lines.push(`\n_model: ${entry.model}_`);
    entry.tool_calls.forEach((tc) => {
      lines.push(`\n**Tool call:** \`${tc.tool}\`(${JSON.stringify(tc.arguments || {})}) -> ${tc.result}`);
    });
    entry.think_blocks.forEach((tb) => {
      lines.push(`\n<details><summary>Thinking</summary>\n\n${tb}\n\n</details>`);
    });
    if (entry.debug) {
      lines.push('\n<details><summary>Raw debug</summary>\n\n```json\n' + JSON.stringify(entry.debug, null, 2) + '\n```\n\n</details>');
    }
    lines.push('');
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chitragupt-debug-${Date.now()}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const MAX_FRAME_DIM = 1024;

function scaledDims(w, h, maxDim) {
  const scale = Math.min(1, maxDim / Math.max(w, h));
  return { width: w * scale, height: h * scale };
}

// ─── Voice input ────────────────────────────────────────────────────────────
const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let isRecording = false;

function initVoiceInput() {
  const micBtn = document.getElementById('mic-btn');
  if (!SpeechRecognitionImpl) return;
  micBtn.style.display = '';

  recognizer = new SpeechRecognitionImpl();
  recognizer.lang = 'en-US';
  recognizer.continuous = false;
  recognizer.interimResults = true;

  recognizer.onresult = (event) => {
    let transcript = '';
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    document.getElementById('prompt-input').value = transcript;
  };

  recognizer.onerror = () => stopVoiceInput();

  recognizer.onend = () => {
    isRecording = false;
    micBtn.classList.remove('recording');
    const text = document.getElementById('prompt-input').value.trim();
    if (text) sendMessage();
  };
}

function toggleVoiceInput() {
  if (!recognizer || isProcessing) return;
  const micBtn = document.getElementById('mic-btn');
  if (isRecording) {
    recognizer.stop();
    return;
  }
  document.getElementById('prompt-input').value = '';
  isRecording = true;
  micBtn.classList.add('recording');
  try {
    recognizer.start();
  } catch {
    isRecording = false;
    micBtn.classList.remove('recording');
  }
}

initVoiceInput();

async function checkHealth() {
  try {
    const resp = await fetch('/health');
    if (resp.ok) document.getElementById('status-text').textContent = 'Connected';
  } catch { /* ignore */ }
}
checkHealth();

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function (e) {
    const img = new Image();
    img.onload = function () {
      const { width, height } = scaledDims(img.naturalWidth, img.naturalHeight, MAX_FRAME_DIM);
      const canvas = document.getElementById('capture-canvas');
      canvas.width = width;
      canvas.height = height;
      canvas.getContext('2d').drawImage(img, 0, 0, width, height);
      const resizedDataUrl = canvas.toDataURL('image/jpeg', 0.85);
      currentImageBase64 = resizedDataUrl.split(',')[1];
      document.getElementById('preview-img').src = resizedDataUrl;
      document.getElementById('image-preview').style.display = 'flex';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentImageBase64 = null;
  document.getElementById('image-preview').style.display = 'none';
  document.getElementById('preview-img').src = '';
  document.getElementById('file-input').value = '';
}

// ─── Activity — same signal as the normal app's sidebar log, just appended
// straight into the message stream since there's no sidebar to put it in.
// Not capped: the whole point of this page is not losing anything.
function setActivityStatus(text) {
  const el = document.getElementById('activity-status');
  if (el) el.textContent = text;
}

function logActivity(thumbDataUrl, text, status) {
  const container = document.getElementById('messages');
  const entry = document.createElement('div');
  entry.className = 'activity-entry status-' + status;
  const time = new Date().toLocaleTimeString([], { hour12: false });
  entry.innerHTML =
    (thumbDataUrl ? '<img src="data:image/jpeg;base64,' + thumbDataUrl + '">' : '') +
    '<div class="activity-text"><span>' + text + '</span><span class="activity-time">' + time + '</span></div>';
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
  return entry;
}

function updateActivityEntry(entry, status, text) {
  if (!entry) return;
  entry.className = 'activity-entry status-' + status;
  entry.querySelector('.activity-text span').textContent = text;
}

function flashCameraFrame() {
  const preview = document.getElementById('live-preview');
  preview.classList.add('frame-flash');
  setTimeout(() => preview.classList.remove('frame-flash'), 200);
}

function addDebugMessage(text, kind) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message debug' + (kind ? ' debug-' + kind : '');
  const time = new Date().toLocaleTimeString([], { hour12: false });
  div.textContent = '[' + time + '] ' + text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// extras.rawData (the full response payload) + extras.requestObj (what was
// POSTed) trigger a full raw dump under the bubble. Plain notices (mode
// switches, resets) just omit those and render as a normal message.
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
      html += '<div class="tool-msg">⚡ Used tool: ' + tc.tool + '(' + JSON.stringify(tc.arguments || {}) + ')</div>';
    });
  }
  div.innerHTML = html;
  if (extras && extras.rawData) {
    renderDebugDump(div, extras.requestObj || {}, extras.rawData);
  }
  logTranscript(role, content, extras);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function captureCurrentFrame() {
  const video = document.getElementById('camera-video');
  if (!cameraStreamActive || !video || !video.videoWidth) return null;
  const { width, height } = scaledDims(video.videoWidth, video.videoHeight, MAX_FRAME_DIM);
  const canvas = document.getElementById('capture-canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(video, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
}

function addCameraEnableMessage(onEnabled) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = "I'd need the camera to check that.<br>";
  const btn = document.createElement('button');
  btn.className = 'camera-enable-btn';
  btn.textContent = '🎥 Enable camera';
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Starting camera...';
    const ok = await startCameraStream();
    if (ok) {
      div.remove();
      onEnabled();
    } else {
      btn.disabled = false;
      btn.textContent = '🎥 Enable camera';
    }
  };
  div.appendChild(btn);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// Live-updating bubble for the streaming endpoint. Raw dump is attached once
// the "done" event's full data arrives (finalize()), same as the plain path.
function createLiveMessage() {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant streaming';

  const thinkEl = document.createElement('details');
  thinkEl.className = 'live-thinking';
  thinkEl.style.display = 'none';
  thinkEl.innerHTML = '<summary>💭 Thinking…</summary><div class="think-body"></div>';
  const thinkBody = thinkEl.querySelector('.think-body');

  const toolsEl = document.createElement('div');
  toolsEl.className = 'live-tools';

  const textEl = document.createElement('div');
  textEl.className = 'live-text';

  div.appendChild(thinkEl);
  div.appendChild(toolsEl);
  div.appendChild(textEl);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;

  const toolLines = new Map();

  return {
    el: div,
    onReasoning(text) {
      thinkEl.style.display = '';
      thinkBody.textContent += text;
      container.scrollTop = container.scrollHeight;
    },
    onContent(text) {
      textEl.textContent += text;
      container.scrollTop = container.scrollHeight;
    },
    onToolStart(name) {
      const line = document.createElement('div');
      line.className = 'tool-msg tool-pending';
      line.textContent = '⚡ Calling ' + name + '...';
      toolsEl.appendChild(line);
      toolLines.set(name, line);
      container.scrollTop = container.scrollHeight;
    },
    onToolResult(name) {
      const line = toolLines.get(name);
      if (line) {
        line.className = 'tool-msg';
        line.textContent = '⚡ Used tool: ' + name;
      }
    },
    finalize(data, requestObj) {
      div.classList.remove('streaming');
      thinkEl.querySelector('summary').textContent = '💭 Thinking';
      textEl.textContent = data.text || '...';
      if (data.model || data.provider) {
        const tag = document.createElement('div');
        tag.className = 'model-tag';
        tag.textContent = (data.provider || '') + '/' + (data.model || '');
        div.appendChild(tag);
      }
      renderDebugDump(div, requestObj || {}, data);
      logTranscript('assistant', data.text || '', {
        model: data.model ? (data.provider || '') + '/' + data.model : null,
        tool_calls: data.tool_calls || [],
        think_blocks: data.think_blocks || [],
        rawData: data,
      });
      container.scrollTop = container.scrollHeight;
    },
    fail(message) {
      div.classList.remove('streaming');
      textEl.textContent = '⚠️ Error: ' + message;
    },
    remove() { div.remove(); },
  };
}

async function runStreamedTurn(prompt, imageBase64, isCameraFollowup) {
  const live = createLiveMessage();
  const requestObj = {
    prompt,
    is_camera_followup: !!isCameraFollowup,
    has_image: !!imageBase64,
    image_base64: imageBase64 || null,
  };
  addDebugMessage(
    'POST /v1/chat/stream  is_camera_followup=' + !!isCameraFollowup +
    '  has_image=' + !!imageBase64 + '  prompt="' + prompt.slice(0, 60) + (prompt.length > 60 ? '…' : '') + '"',
    'send',
  );
  const resp = await fetch('/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      image_base64: imageBase64,
      is_camera_followup: !!isCameraFollowup,
    }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalData = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split('\n').find((l) => l.startsWith('data: '));
      if (!line) continue;
      const event = JSON.parse(line.slice(6));
      switch (event.type) {
        case 'tool_call_start': addDebugMessage('⚡ tool_call_start: ' + event.name, 'recv'); break;
        case 'tool_result': addDebugMessage('⚡ tool_result: ' + event.tool + ' -> ' + String(event.result).slice(0, 300), 'recv'); break;
        case 'error': addDebugMessage('✖ stream error: ' + event.message, 'error'); break;
      }
      switch (event.type) {
        case 'reasoning_delta': live.onReasoning(event.text); break;
        case 'content_delta': live.onContent(event.text); break;
        case 'tool_call_start': live.onToolStart(event.name); break;
        case 'tool_result': live.onToolResult(event.tool); break;
        case 'done': finalData = event.data; break;
        case 'error': throw new Error(event.message);
      }
    }
  }

  if (finalData) {
    live.finalize(finalData, requestObj);
    addDebugMessage(
      '← done  provider=' + finalData.provider + ' model=' + finalData.model +
      '  tool_calls=[' + (finalData.tool_calls || []).map(t => t.tool).join(',') + ']' +
      '  needs_camera=' + !!finalData.needs_camera + '  needs_live_search=' + !!finalData.needs_live_search,
      'recv',
    );
  } else {
    live.fail('stream ended with no response');
    addDebugMessage('✖ stream ended with no "done" event', 'error');
  }
  return finalData;
}

function setSending(active) {
  isProcessing = active;
  document.getElementById('send-btn').disabled = active;
}

async function sendMessage() {
  if (isProcessing) return;
  const input = document.getElementById('prompt-input');
  const prompt = input.value.trim();
  if (!prompt && !currentImageBase64) return;

  setSending(true);
  addMessage('user', prompt || '(image uploaded)', {});
  input.value = '';
  setActivityStatus('📤 Sending message to API…');

  try {
    let data = await runStreamedTurn(prompt || 'What do you see in this image?', currentImageBase64);

    if (data && data.needs_camera) {
      const frame = captureCurrentFrame();
      if (frame) {
        setActivityStatus('📤 Sending requested camera frame to API…');
        logActivity(frame, 'Camera frame sent (model requested it)', 'sending');
        data = await runStreamedTurn(prompt || 'What do you see in this image?', frame, true);
      } else {
        clearImage();
        setSending(false);
        setActivityStatus('Idle');
        addCameraEnableMessage(async () => {
          setSending(true);
          try {
            const retryFrame = captureCurrentFrame();
            setActivityStatus('📤 Sending requested camera frame to API…');
            logActivity(retryFrame, 'Camera frame sent (model requested it)', 'sending');
            await runStreamedTurn(prompt || 'What do you see in this image?', retryFrame, true);
          } catch (err) {
            addMessage('assistant', '⚠️ Error: ' + err.message, {});
          }
          setSending(false);
          setActivityStatus('Idle');
          input.focus();
        });
        return;
      }
    }

    clearImage();

    if (data && data.needs_live_search) {
      await switchMode('live');
    }
  } catch (err) {
    addMessage('assistant', '⚠️ Error: ' + err.message, {});
  }

  setSending(false);
  setActivityStatus('Idle');
  input.focus();
}

async function resetConversation() {
  await fetch('/v1/reset', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
  transcriptLog = [];
  addMessage('assistant', 'Conversation reset. How can I help you?', {});
}

// ─── Live camera streaming ─────────────────────────────────────────────────

const LIVE_SETTINGS_KEY = 'chitragupt-debug-live-settings';
const THRESHOLD_LEVELS = { 1: 6, 2: 12, 3: 22 };
const THRESHOLD_LABELS = { 1: 'high', 2: 'medium', 3: 'low' };

let liveActive = false;
let cameraStreamActive = false;
let liveStream = null;
let liveTimer = null;
let liveSending = false;
let lastSentDiffData = null;
let framesWatched = 0;
let framesSent = 0;

function loadLiveSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(LIVE_SETTINGS_KEY) || '{}');
    if (saved.intervalS) document.getElementById('interval-slider').value = saved.intervalS;
    if (saved.thresholdLevel) document.getElementById('threshold-slider').value = saved.thresholdLevel;
  } catch { /* ignore malformed settings */ }
  updateSettingsLabels();
}

function saveLiveSettings() {
  localStorage.setItem(LIVE_SETTINGS_KEY, JSON.stringify({
    intervalS: document.getElementById('interval-slider').value,
    thresholdLevel: document.getElementById('threshold-slider').value,
  }));
}

function updateSettingsLabels() {
  document.getElementById('interval-value').textContent = document.getElementById('interval-slider').value;
  const level = document.getElementById('threshold-slider').value;
  document.getElementById('threshold-value').textContent = THRESHOLD_LABELS[level];
}

function updateLiveStats() {
  const el = document.getElementById('live-stats');
  if (!liveActive) {
    el.textContent = 'not watching';
    return;
  }
  el.textContent = `watching — ${framesWatched} sampled, ${framesSent} sent`;
}

document.addEventListener('DOMContentLoaded', () => {
  loadLiveSettings();
  document.getElementById('interval-slider').addEventListener('input', () => { updateSettingsLabels(); saveLiveSettings(); restartLiveTimer(); });
  document.getElementById('threshold-slider').addEventListener('input', () => { updateSettingsLabels(); saveLiveSettings(); });
  startTimerPolling();
});

// ─── Background timers ──────────────────────────────────────────────────────

const TIMER_POLL_INTERVAL_S = 15;

function startTimerPolling() {
  checkTimers();
  setInterval(checkTimers, TIMER_POLL_INTERVAL_S * 1000);
}

async function checkTimers() {
  try {
    const resp = await fetch('/v1/timers/check');
    const data = await resp.json();
    (data.completed || []).forEach((t) => {
      addMessage('assistant', `⏰ ${t.label}: ${t.message}`, { rawData: { debug: t.debug, text: t.message, model: t.debug && t.debug.steps && t.debug.steps[0] && t.debug.steps[0].model, provider: t.debug && t.debug.steps && t.debug.steps[0] && t.debug.steps[0].provider, tool_calls: (t.debug && t.debug.tool_calls) || [] }, requestObj: { poll: '/v1/timers/check', timer_id: t.id, label: t.label } });
    });
  } catch { /* ignore — next poll will retry */ }
}

// ─── Mode switching ─────────────────────────────────────────────────────────

let currentMode = 'chat';

async function switchMode(mode) {
  if (mode === currentMode) return;

  if (mode === 'live') {
    await startLive();
    if (!liveActive) return;
  } else if (currentMode === 'live') {
    stopLive();
  }

  currentMode = mode;
  document.getElementById('mode-chat-btn').classList.toggle('active', mode === 'chat');
  document.getElementById('mode-live-btn').classList.toggle('active', mode === 'live');
  document.getElementById('upload-img-btn').style.display = mode === 'live' ? 'none' : '';
  document.getElementById('camera-toggle-btn').style.display = mode === 'live' ? 'none' : '';
  document.getElementById('prompt-input').placeholder =
    mode === 'live' ? 'Ask about what the camera sees (optional)...' : 'Ask me anything...';
}

async function startCameraStream() {
  if (cameraStreamActive) return true;
  if (!window.isSecureContext) {
    addMessage('assistant', '⚠️ Camera requires a secure connection (HTTPS). Deploy behind HTTPS or use localhost to test.', {});
    return false;
  }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
  } catch (err) {
    addMessage('assistant', '⚠️ Could not access camera: ' + err.message, {});
    return false;
  }
  const video = document.getElementById('camera-video');
  video.srcObject = liveStream;
  document.getElementById('live-preview').style.display = 'flex';
  cameraStreamActive = true;
  updateCameraToggleBtn();
  return true;
}

function stopCameraStream() {
  if (liveStream) {
    liveStream.getTracks().forEach((t) => t.stop());
    liveStream = null;
  }
  document.getElementById('live-preview').style.display = 'none';
  cameraStreamActive = false;
  updateCameraToggleBtn();
}

function updateCameraToggleBtn() {
  const btn = document.getElementById('camera-toggle-btn');
  if (!btn) return;
  btn.classList.toggle('camera-on', cameraStreamActive);
  btn.title = cameraStreamActive ? 'Turn camera off' : 'Enable camera for on-demand questions';
}

async function toggleCameraStream() {
  if (cameraStreamActive && !liveActive) {
    stopCameraStream();
  } else if (!cameraStreamActive) {
    await startCameraStream();
  }
}

async function startLive() {
  const ok = await startCameraStream();
  if (!ok) return;

  document.getElementById('mode-live-btn').classList.add('live-active');
  liveActive = true;
  framesWatched = 0;
  framesSent = 0;
  lastSentDiffData = null;
  updateLiveStats();
  restartLiveTimer();
  setActivityStatus('👁 Watching for changes…');
  addMessage('assistant', '📹 Live mode on — watching for changes, sending at most once per interval.', {});
}

function stopLive() {
  liveActive = false;
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  document.getElementById('mode-live-btn').classList.remove('live-active');
  updateLiveStats();
  stopCameraStream();
  setActivityStatus('Idle');
}

function restartLiveTimer() {
  if (!liveActive) return;
  if (liveTimer) clearInterval(liveTimer);
  const intervalS = Number(document.getElementById('interval-slider').value);
  liveTimer = setInterval(sampleLiveFrame, intervalS * 1000);
}

function captureDiffSample(video) {
  const diffCanvas = document.getElementById('diff-canvas');
  const ctx = diffCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, diffCanvas.width, diffCanvas.height);
  return ctx.getImageData(0, 0, diffCanvas.width, diffCanvas.height).data;
}

function meanGrayscaleDelta(a, b) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    const grayA = (a[i] + a[i + 1] + a[i + 2]) / 3;
    const grayB = (b[i] + b[i + 1] + b[i + 2]) / 3;
    total += Math.abs(grayA - grayB);
    count += 1;
  }
  return total / count;
}

let pendingLiveFrame = null;

async function sampleLiveFrame() {
  if (!liveActive) return;
  const video = document.getElementById('camera-video');
  if (!video.videoWidth) return;

  framesWatched += 1;
  const sample = captureDiffSample(video);

  const hasTypedPrompt = !!document.getElementById('prompt-input').value.trim();

  if (!hasTypedPrompt && lastSentDiffData) {
    const threshold = THRESHOLD_LEVELS[document.getElementById('threshold-slider').value];
    const delta = meanGrayscaleDelta(sample, lastSentDiffData);
    if (delta < threshold) {
      updateLiveStats();
      setActivityStatus('👁 Watching — scene unchanged, not sent (delta=' + delta.toFixed(1) + ' < ' + threshold + ')');
      return;
    }
  }

  lastSentDiffData = sample;
  framesSent += 1;
  updateLiveStats();

  if (liveSending) {
    pendingLiveFrame = video;
    return;
  }
  await sendLiveFrame(video);
}

async function sendLiveFrame(video) {
  liveSending = true;
  const canvas = document.getElementById('capture-canvas');
  const { width, height } = scaledDims(video.videoWidth, video.videoHeight, MAX_FRAME_DIM);
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageBase64 = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];

  const input = document.getElementById('prompt-input');
  const typedPrompt = input.value.trim();
  const prompt = typedPrompt || 'Watch tick — check the scene against the active task, if any; stay silent if nothing relevant changed.';
  if (typedPrompt) input.value = '';

  flashCameraFrame();
  setActivityStatus('📤 Sent frame #' + framesSent + ' — awaiting response…');
  const entry = logActivity(imageBase64, 'Frame #' + framesSent + ' sent to API', 'sending');
  const requestObj = { prompt, is_live_frame: !typedPrompt, has_image: true, image_base64: imageBase64 };
  addDebugMessage(
    'POST /v1/chat  frame #' + framesSent + '  is_live_frame=' + !typedPrompt +
    '  prompt="' + prompt.slice(0, 60) + (prompt.length > 60 ? '…' : '') + '"',
    'send',
  );

  try {
    const resp = await fetch('/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        image_base64: imageBase64,
        is_live_frame: !typedPrompt,
      }),
    });
    const data = await resp.json();

    if (data.vision_prompt || data.scene_description) {
      addDebugMessage('  [vision→Qwen] asked: "' + (data.vision_prompt || '') + '"', 'send');
      addDebugMessage('  [vision←Qwen] said: "' + (data.scene_description || '') + '"', 'recv');
    }
    if (data.rate_limited) {
      const waitS = data.retry_after || 5;
      updateActivityEntry(entry, 'silent', 'Frame #' + framesSent + ' — rate limited, pausing ' + waitS.toFixed(1) + 's');
      addDebugMessage('← 429 rate_limited  retry_after=' + waitS + 's  pausing live polling', 'rate');
      addMessage('assistant', '(rate limited, tick skipped)', { rawData: data, requestObj });
      setActivityStatus('⏳ Rate limited — pausing ' + Math.ceil(waitS) + 's…');
      if (liveActive && liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
        setTimeout(() => { if (liveActive) restartLiveTimer(); }, waitS * 1000);
      }
    } else if (!data.scene_unchanged && data.text) {
      addMessage('assistant', data.text, { model: data.provider + '/' + data.model, tool_calls: data.tool_calls, think_blocks: data.think_blocks, rawData: data, requestObj });
      updateActivityEntry(entry, 'replied', 'Frame #' + framesSent + ' — model replied');
      addDebugMessage(
        '← 200  provider=' + data.provider + ' model=' + data.model +
        '  tool_calls=[' + (data.tool_calls || []).map(t => t.tool).join(',') + ']',
        'recv',
      );
    } else {
      updateActivityEntry(entry, 'silent', 'Frame #' + framesSent + ' — silent (no relevant change)');
      // Still dump the raw pipeline data for a silent tick — "silent" is a
      // real outcome worth inspecting (was it [SILENT], or truly empty?).
      addMessage('assistant', '(silent — no visible reply this tick)', { rawData: data, requestObj });
      addDebugMessage(
        '← 200  provider=' + (data.provider || 'n/a') + ' model=' + (data.model || 'n/a') +
        '  scene_unchanged=' + !!data.scene_unchanged + '  silent=' + !data.text,
        'silent',
      );
    }
  } catch (err) {
    updateActivityEntry(entry, 'error', 'Frame #' + framesSent + ' — error: ' + err.message);
    addDebugMessage('✖ request failed: ' + err.message, 'error');
    addMessage('assistant', '⚠️ Live frame error: ' + err.message, {});
  } finally {
    liveSending = false;
    if (pendingLiveFrame && liveActive) {
      const frame = pendingLiveFrame;
      pendingLiveFrame = null;
      sendLiveFrame(frame);
    } else {
      pendingLiveFrame = null;
      if (liveActive) setActivityStatus('👁 Watching for changes…');
    }
  }
}
