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

// Captures the current live camera frame, if the stream is actually
// running — used to fulfill a needs_camera round trip from a text-only turn.
function captureCurrentFrame() {
  const video = document.getElementById('camera-video');
  if (!liveActive || !video || !video.videoWidth) return null;
  const canvas = document.getElementById('capture-canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
}

async function postChat(prompt, imageBase64) {
  const resp = await fetch('/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, image_base64: imageBase64 }),
  });
  return resp.json();
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
    let data = await postChat(prompt || 'What do you see in this image?', currentImageBase64);

    // The model asked to see the current scene instead of guessing — grab
    // a fresh frame and resend the same question once. No image available
    // (camera not on) means we can't fulfill it; say so instead of looping.
    if (data.needs_camera) {
      const frame = captureCurrentFrame();
      if (frame) {
        data = await postChat(prompt || 'What do you see in this image?', frame);
      } else {
        hideTyping();
        addMessage('assistant', "I'd need the camera on to check that — open Live Watch first.", {});
        clearImage();
        isProcessing = false;
        document.getElementById('send-btn').disabled = false;
        return;
      }
    }

    hideTyping();

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

let liveActive = false;
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
  document.getElementById('prompt-input').placeholder =
    mode === 'live' ? 'Ask about what the camera sees (optional)...' : 'Ask me anything...';
}

async function startLive() {
  if (!window.isSecureContext) {
    addMessage('assistant', '⚠️ Live camera requires a secure connection (HTTPS). Deploy behind HTTPS or use localhost to test.', {});
    return;
  }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
  } catch (err) {
    addMessage('assistant', '⚠️ Could not access camera: ' + err.message, {});
    return;
  }

  const video = document.getElementById('camera-video');
  video.srcObject = liveStream;
  document.getElementById('live-preview').style.display = 'flex';
  document.getElementById('mode-live-btn').classList.add('live-active');

  liveActive = true;
  framesWatched = 0;
  framesSent = 0;
  lastSentDiffData = null;
  updateLiveStats();
  restartLiveTimer();
  addMessage('assistant', '📹 Live mode on — watching for changes, sending at most once per interval.', {});
}

function stopLive() {
  liveActive = false;
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  if (liveStream) {
    liveStream.getTracks().forEach((t) => t.stop());
    liveStream = null;
  }
  document.getElementById('live-preview').style.display = 'none';
  document.getElementById('mode-live-btn').classList.remove('live-active');
  updateLiveStats();
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

  if (lastSentDiffData) {
    const threshold = THRESHOLD_LEVELS[document.getElementById('threshold-slider').value];
    const delta = meanGrayscaleDelta(sample, lastSentDiffData);
    if (delta < threshold) {
      updateLiveStats();
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

const MAX_FRAME_DIM = 1024; // cap resolution to save vision tokens; keep JPEG quality high so labels/text stay readable

async function sendLiveFrame(video) {
  liveSending = true;
  const canvas = document.getElementById('capture-canvas');
  const scale = Math.min(1, MAX_FRAME_DIM / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = video.videoWidth * scale;
  canvas.height = video.videoHeight * scale;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageBase64 = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];

  const input = document.getElementById('prompt-input');
  const typedPrompt = input.value.trim();
  const prompt = typedPrompt || 'Watch tick — check the scene against the active task, if any; stay silent if nothing relevant changed.';
  if (typedPrompt) input.value = '';

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
    if (!data.scene_unchanged && data.text) {
      let displayText = data.text;
      if (data.think_blocks && data.think_blocks.length > 0) {
        displayText += '\n\n<details><summary>💭 Thinking</summary>\n' + data.think_blocks.join('\n') + '\n</details>';
      }
      addMessage('assistant', displayText, { model: data.provider + '/' + data.model, tool_calls: data.tool_calls });
    }
  } catch (err) {
    addMessage('assistant', '⚠️ Live frame error: ' + err.message, {});
  } finally {
    liveSending = false;
    if (pendingLiveFrame && liveActive) {
      const frame = pendingLiveFrame;
      pendingLiveFrame = null;
      sendLiveFrame(frame); // flush immediately rather than waiting for the next interval tick
    } else {
      pendingLiveFrame = null;
    }
  }
}
