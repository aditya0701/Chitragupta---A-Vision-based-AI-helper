let currentImageBase64 = null;
let isProcessing = false;

// ─── Conversation export ────────────────────────────────────────────────────
// Mirrors every rendered message (plain notices via addMessage, and streamed
// turns via createLiveMessage().finalize) into a plain-data log, independent
// of the DOM — so "Save conversation" can dump a clean transcript (including
// think blocks and tool calls) for pasting into a bug report, without having
// to scrape rendered HTML back out.
let transcriptLog = [];

function logTranscript(role, text, extras) {
  transcriptLog.push({
    role,
    text,
    model: extras && extras.model,
    tool_calls: (extras && extras.tool_calls) || [],
    think_blocks: (extras && extras.think_blocks) || [],
    at: new Date().toISOString(),
  });
}

function exportConversation() {
  const lines = ['# Chitragupt conversation export', `Exported ${new Date().toISOString()}`, ''];
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
    lines.push('');
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chitragupt-conversation-${Date.now()}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Shared resolution cap for every image sent to the backend (upload, live
// tick, or a one-off camera capture) — keeps vision-token cost down without
// a visible quality hit, since JPEG quality stays high (see below).
const MAX_FRAME_DIM = 1024;

function scaledDims(w, h, maxDim) {
  const scale = Math.min(1, maxDim / Math.max(w, h));
  return { width: w * scale, height: h * scale };
}

// ─── Voice input (Web Speech API — browser-native, no server change) ────────
// Only Chrome/Edge/Safari implement SpeechRecognition (Firefox doesn't), and
// it requires a secure context (HTTPS or localhost) same as getUserMedia —
// the mic button stays hidden entirely rather than showing a dead control.
const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let isRecording = false;

function initVoiceInput() {
  const micBtn = document.getElementById('mic-btn');
  if (!SpeechRecognitionImpl) return; // leave button hidden — no support
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

  // Fires when the browser decides you've stopped talking (continuous=false)
  // — send automatically so speaking a question is the whole interaction,
  // no follow-up tap needed.
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
    if (resp.ok) {
      document.getElementById('status-dot').className = 'status-dot';
      document.getElementById('status-text').textContent = 'Connected';
    }
  } catch { /* ignore */ }
}
checkHealth();

function toggleSidebar(force) {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  const open = force !== undefined ? force : !sidebar.classList.contains('open');
  sidebar.classList.toggle('open', open);
  backdrop.classList.toggle('open', open);
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = function() {
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

// ─── Activity log — surfaces exactly what the pipeline is doing right now.
// Added because Live Watch's plain (non-streamed) fetch gave zero feedback
// between "frame captured" and "response arrived," which could be several
// seconds of apparent nothing — especially when the model stays silent
// (the [SILENT] protocol). This gives a live status line plus a per-frame
// log with a thumbnail of the actual frame that went out, so it's clear
// which frame the model is looking at and what's happening to it.
const MAX_ACTIVITY_ENTRIES = 12;

function setActivityStatus(text, busy) {
  const el = document.getElementById('activity-status');
  el.textContent = text;
  el.classList.toggle('busy', !!busy);
}

function logActivity(thumbDataUrl, text, status) {
  const log = document.getElementById('activity-log');
  const entry = document.createElement('div');
  entry.className = 'activity-entry status-' + status;
  const time = new Date().toLocaleTimeString([], { hour12: false });
  entry.innerHTML =
    (thumbDataUrl ? '<img src="data:image/jpeg;base64,' + thumbDataUrl + '">' : '') +
    '<div class="activity-text"><span>' + text + '</span><span class="activity-time">' + time + '</span></div>';
  log.insertBefore(entry, log.firstChild);
  while (log.children.length > MAX_ACTIVITY_ENTRIES) log.removeChild(log.lastChild);
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
  logTranscript(role, content, extras);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// Captures the current camera frame, if a stream is attached — works
// whether the stream came from Live Watch (continuous polling) or the
// manual "Enable camera" toggle (stream only, no autonomous ticking).
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

// A clickable "enable camera and retry" prompt, shown when request_camera
// fires but no stream is attached — replaces a dead-end text message with
// something the user can actually act on (was previously just "open Live
// Watch first," which is not a popup and easy to miss/ignore).
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

// Builds a live-updating assistant message bubble that fills in as
// reasoning_delta/content_delta/tool_call_start/tool_result events arrive
// from /v1/chat/stream, instead of appearing all at once at the end —
// mirrors how tool calls show up mid-generation in chat apps like Claude
// Code, rather than only in the final dumped response.
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
    finalize(data) {
      div.classList.remove('streaming');
      thinkEl.querySelector('summary').textContent = '💭 Thinking';
      textEl.textContent = data.text || '...';
      if (data.model || data.provider) {
        const tag = document.createElement('div');
        tag.className = 'model-tag';
        tag.textContent = (data.provider || '') + '/' + (data.model || '');
        div.appendChild(tag);
      }
      logTranscript('assistant', data.text || '', {
        model: data.model ? (data.provider || '') + '/' + data.model : null,
        tool_calls: data.tool_calls || [],
        think_blocks: data.think_blocks || [],
      });
    },
    fail(message) {
      div.classList.remove('streaming');
      textEl.textContent = '⚠️ Error: ' + message;
    },
    remove() { div.remove(); },
  };
}

// Posts to the streaming endpoint and parses the SSE frames (data: {...}\n\n)
// by hand — fetch()'s ReadableStream instead of EventSource, since
// EventSource only supports GET and this needs to POST the prompt/image.
// Wires each event straight into the live bubble as it arrives; returns the
// final "done" event's data once the stream ends.
async function runStreamedTurn(prompt, imageBase64, isCameraFollowup) {
  const live = createLiveMessage();
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
    live.finalize(finalData);
  } else {
    live.fail('stream ended with no response');
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
  setActivityStatus('📤 Sending message to API…', true);

  try {
    let data = await runStreamedTurn(prompt || 'What do you see in this image?', currentImageBase64);

    // The model asked to see the current scene instead of guessing — grab
    // a fresh frame and resend the same question once. No stream attached
    // (camera never enabled) means we can't fulfill it yet — offer a
    // one-click way to turn the camera on and retry, instead of a dead-end
    // message the user has no way to act on.
    if (data && data.needs_camera) {
      const frame = captureCurrentFrame();
      if (frame) {
        setActivityStatus('📤 Sending requested camera frame to API…', true);
        logActivity(frame, 'Camera frame sent (model requested it)', 'sending');
        data = await runStreamedTurn(prompt || 'What do you see in this image?', frame, true);
      } else {
        clearImage();
        setSending(false);
        setActivityStatus('Idle', false);
        addCameraEnableMessage(async () => {
          setSending(true);
          try {
            const retryFrame = captureCurrentFrame();
            setActivityStatus('📤 Sending requested camera frame to API…', true);
            logActivity(retryFrame, 'Camera frame sent (model requested it)', 'sending');
            await runStreamedTurn(prompt || 'What do you see in this image?', retryFrame, true);
          } catch (err) {
            addMessage('assistant', '⚠️ Error: ' + err.message, {});
          }
          setSending(false);
          setActivityStatus('Idle', false);
          input.focus();
        });
        return;
      }
    }

    clearImage();

    // The model wants to keep watching for something specific, not just
    // take one look — switch into Live Watch so it actually can. If
    // already there (or the camera's already on), this is a no-op.
    if (data && data.needs_live_search) {
      await switchMode('live');
    }
  } catch (err) {
    addMessage('assistant', '⚠️ Error: ' + err.message, {});
  }

  setSending(false);
  setActivityStatus('Idle', false);
  input.focus();
}

async function resetConversation() {
  await fetch('/v1/reset', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
  transcriptLog = [];
  addMessage('assistant', 'Conversation reset. How can I help you?', {});
  toggleSidebar(false);
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => { /* ignore */ });
  });
}

// ─── Live camera streaming (budget-conscious) ─────────────────────────────
//
// Two throttles run client-side, before any network call is made:
//   1. A fixed sampling interval bounds the worst-case request rate.
//   2. A cheap perceptual diff against the last *sent* frame skips the
//      network/Gemini call entirely when the scene hasn't meaningfully
//      changed. This is what actually protects the free-tier quota, since
//      it runs before the request leaves the browser.

const LIVE_SETTINGS_KEY = 'chitragupt-live-settings';
// Diff thresholds (mean grayscale delta on a 32x32 downsample, 0-255 scale).
// Lower = more sensitive (sends more often). Slider maps 1/2/3 -> these.
const THRESHOLD_LEVELS = { 1: 6, 2: 12, 3: 22 };
const THRESHOLD_LABELS = { 1: 'high', 2: 'medium', 3: 'low' };

let liveActive = false; // Live Watch's autonomous polling loop specifically
let cameraStreamActive = false; // camera stream attached at all (Live Watch OR manual toggle)
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
    el.textContent = 'Not watching';
    return;
  }
  el.textContent = `Watching — ${framesWatched} sampled, ${framesSent} sent`;
}

document.addEventListener('DOMContentLoaded', () => {
  loadLiveSettings();
  document.getElementById('interval-slider').addEventListener('input', () => { updateSettingsLabels(); saveLiveSettings(); restartLiveTimer(); });
  document.getElementById('threshold-slider').addEventListener('input', () => { updateSettingsLabels(); saveLiveSettings(); });
  startTimerPolling();
});

// ─── Background timers (cooking steps, wait periods) ──────────────────────
//
// Polls a cheap server endpoint that's pure arithmetic unless a timer has
// actually completed — no Groq cost per poll, only once per fired timer.

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
      addMessage('assistant', `⏰ ${t.label}: ${t.message}`, {});
    });
  } catch { /* ignore — next poll will retry */ }
}

// ─── Mode switching (Chat & Image vs Live Watch) ───────────────────────────

let currentMode = 'chat';

async function switchMode(mode) {
  if (mode === currentMode) return;

  if (mode === 'live') {
    await startLive();
    if (!liveActive) return; // camera permission denied / unavailable — stay on chat mode
  } else if (currentMode === 'live') {
    stopLive();
  }

  currentMode = mode;
  document.getElementById('mode-chat-btn').classList.toggle('active', mode === 'chat');
  document.getElementById('mode-live-btn').classList.toggle('active', mode === 'live');
  document.getElementById('upload-img-btn').style.display = mode === 'live' ? 'none' : '';
  // Live Watch owns the camera lifecycle directly while active — the manual
  // toggle is only for on-demand capture in Chat mode, showing both would
  // just invite confusion about which one is actually in control.
  document.getElementById('camera-toggle-btn').style.display = mode === 'live' ? 'none' : '';
  document.getElementById('prompt-input').placeholder =
    mode === 'live' ? 'Ask about what the camera sees (optional)...' : 'Ask me anything...';
}

// Raw camera stream lifecycle — just getUserMedia + wiring the <video>
// element, no autonomous polling. Used directly by the manual "Enable
// camera" toggle (on-demand capture only, no per-interval Groq calls), and
// as the first step of full Live Watch mode (see startLive below).
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
  btn.title = cameraStreamActive ? 'Turn camera off' : 'Enable camera for on-demand questions (no continuous watching)';
}

// Manual toggle — deliberately does NOT touch Live Watch's polling loop.
// This is "camera available for request_camera to use," not "watch
// continuously" — kept separate so the camera is never running (and
// costing nothing extra since it's stream-only, but also never silently
// left on) unless the user explicitly asked for either mode.
async function toggleCameraStream() {
  if (cameraStreamActive && !liveActive) {
    stopCameraStream();
  } else if (!cameraStreamActive) {
    await startCameraStream();
  }
  // If Live Watch's polling is active, leave the stream alone — that tab
  // owns it; use the Live Watch tab's own close button to stop both together.
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
  setActivityStatus('👁 Watching for changes…', false);
  addMessage('assistant', '📹 Live mode on — watching for changes, sending at most once per interval.', {});
}

function stopLive() {
  liveActive = false;
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  document.getElementById('mode-live-btn').classList.remove('live-active');
  updateLiveStats();
  stopCameraStream();
  setActivityStatus('Idle', false);
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

let pendingLiveFrame = null; // latest frame captured while a request was in flight

async function sampleLiveFrame() {
  if (!liveActive) return;
  const video = document.getElementById('camera-video');
  if (!video.videoWidth) return;

  framesWatched += 1;
  const sample = captureDiffSample(video);

  // A typed question waiting to go out always overrides the diff gate — the
  // user is actively waiting on a reply, so an unchanged scene isn't a
  // reason to sit on their message until the next tick happens to clear
  // the threshold (previously this was only checked inside sendLiveFrame,
  // which the diff-gate return below never let it reach).
  const hasTypedPrompt = !!document.getElementById('prompt-input').value.trim();

  if (!hasTypedPrompt && lastSentDiffData) {
    const threshold = THRESHOLD_LEVELS[document.getElementById('threshold-slider').value];
    const delta = meanGrayscaleDelta(sample, lastSentDiffData);
    if (delta < threshold) {
      updateLiveStats();
      setActivityStatus('👁 Watching — scene unchanged, not sent', false);
      return; // scene unchanged enough — skip the network call entirely
    }
  }

  lastSentDiffData = sample;
  framesSent += 1;
  updateLiveStats();

  if (liveSending) {
    // Already mid-request (e.g. a slow tool call) — remember this frame
    // instead of dropping it, so a moment that matters (marinade done, the
    // thing you're looking for comes into view) isn't silently lost while
    // the agent is busy. Only the latest matters; older ones are superseded.
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

  // This is the actual "notify me when a frame is sent" moment — fires the
  // instant the request goes out, not when the response comes back, and
  // logs a thumbnail of the exact frame so it's clear which one the model
  // is now looking at (frame N doesn't map to wall-clock time 1:1 once the
  // diff gate starts skipping ticks).
  flashCameraFrame();
  setActivityStatus('📤 Sent frame #' + framesSent + ' — awaiting response…', true);
  const entry = logActivity(imageBase64, 'Frame #' + framesSent + ' sent to API', 'sending');

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
    if (data.rate_limited) {
      // The server deliberately skipped this tick rather than surfacing a
      // raw 429 — back the polling interval off for the provider's
      // suggested wait instead of hammering the same per-minute cap again
      // next tick. Live Watch resumes at its normal interval afterwards.
      const waitS = data.retry_after || 5;
      updateActivityEntry(entry, 'silent', 'Frame #' + framesSent + ' — rate limited, pausing ' + waitS.toFixed(1) + 's');
      setActivityStatus('⏳ Rate limited — pausing ' + Math.ceil(waitS) + 's…', false);
      if (liveActive && liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
        setTimeout(() => { if (liveActive) restartLiveTimer(); }, waitS * 1000);
      }
    } else if (!data.scene_unchanged && data.text) {
      let displayText = data.text;
      if (data.think_blocks && data.think_blocks.length > 0) {
        displayText += '\n\n<details><summary>💭 Thinking</summary>\n' + data.think_blocks.join('\n') + '\n</details>';
      }
      addMessage('assistant', displayText, { model: data.provider + '/' + data.model, tool_calls: data.tool_calls, think_blocks: data.think_blocks });
      updateActivityEntry(entry, 'replied', 'Frame #' + framesSent + ' — model replied');
    } else {
      updateActivityEntry(entry, 'silent', 'Frame #' + framesSent + ' — silent (no relevant change)');
    }
  } catch (err) {
    updateActivityEntry(entry, 'error', 'Frame #' + framesSent + ' — error: ' + err.message);
    addMessage('assistant', '⚠️ Live frame error: ' + err.message, {});
  } finally {
    liveSending = false;
    if (pendingLiveFrame && liveActive) {
      const frame = pendingLiveFrame;
      pendingLiveFrame = null;
      sendLiveFrame(frame); // flush immediately rather than waiting for the next interval tick
    } else {
      pendingLiveFrame = null;
      if (liveActive) setActivityStatus('👁 Watching for changes…', false);
    }
  }
}
