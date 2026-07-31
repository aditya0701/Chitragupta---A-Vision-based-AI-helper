// Chitragupta Live (v2) — client for the world-doc tick system.
// Independent of app.js; talks only to /v2/*.

const MAX_FRAME_DIM = 1024;
const JPEG_QUALITY = 0.85;
const POLL_INTERVAL_MS = 20000;

let stream = null;
let ticking = false;
let tickTimer = null;
let busy = false;            // a /v2 request is in flight
let pendingFrame = null;     // latest frame captured while busy — flushed when free
let queuedPrompt = null;     // user message spoken/typed while busy — flushed when free
let lastSentFrame = null;    // grayscale sample of the last frame actually sent (diff gate)

const $ = (id) => document.getElementById(id);
const video = $('camera-video');

// ── Transcript ───────────────────────────────────────────────────────────────

function addMsg(kind, text) {
  const div = document.createElement('div');
  div.className = `msg ${kind}`;
  div.textContent = text;
  const ts = document.createElement('span');
  ts.className = 'ts';
  ts.textContent = new Date().toLocaleTimeString();
  div.appendChild(ts);
  $('transcript').appendChild(div);
  $('transcript').scrollTop = $('transcript').scrollHeight;
}

function addCaptionDot(caption) {
  const div = document.createElement('div');
  div.className = $('show-captions').checked ? 'msg system' : 'caption-dot';
  div.textContent = $('show-captions').checked ? `👁 ${caption}` : '·';
  div.title = caption;
  $('transcript').appendChild(div);
  $('transcript').scrollTop = $('transcript').scrollHeight;
}

function setStatus(text) { $('status-line').textContent = text; }

function updateDoc(rendered) {
  if (rendered != null) $('doc-panel').textContent = rendered || '(empty)';
}

// ── Camera ───────────────────────────────────────────────────────────────────

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false,
    });
  } catch (e) {
    addMsg('system', `Camera failed: ${e.message}`);
    return;
  }
  video.srcObject = stream;
  video.style.display = 'block';
  $('camera-off').style.display = 'none';
  $('camera-btn').textContent = '🎥 Stop camera';
  $('camera-btn').classList.add('active');
  $('tick-btn').disabled = false;
}

function stopCamera() {
  stopTicks();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  video.style.display = 'none';
  $('camera-off').style.display = 'block';
  $('camera-btn').textContent = '🎥 Start camera';
  $('camera-btn').classList.remove('active');
  $('tick-btn').disabled = true;
}

function captureFrame() {
  if (!stream || video.videoWidth === 0) return null;
  const scale = Math.min(1, MAX_FRAME_DIM / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];
}

// Cheap perceptual diff: mean abs delta over a 32x32 grayscale sample.
// Runs before any network call — the main API cost control.
function graySample() {
  if (!stream || video.videoWidth === 0) return null;
  const c = document.createElement('canvas');
  c.width = 32; c.height = 32;
  const ctx = c.getContext('2d');
  ctx.drawImage(video, 0, 0, 32, 32);
  const data = ctx.getImageData(0, 0, 32, 32).data;
  const gray = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  return gray;
}

function meanDelta(a, b) {
  if (!a || !b) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// ── Tick loop ────────────────────────────────────────────────────────────────

function startTicks() {
  ticking = true;
  $('tick-btn').textContent = '⏸ Stop ticks';
  $('tick-btn').classList.add('active');
  scheduleTick();
}

function stopTicks() {
  ticking = false;
  clearTimeout(tickTimer);
  $('tick-btn').textContent = '▶ Start ticks';
  $('tick-btn').classList.remove('active');
  setStatus('idle');
}

function scheduleTick() {
  if (!ticking) return;
  clearTimeout(tickTimer);
  tickTimer = setTimeout(onTick, Number($('interval').value) * 1000);
}

async function onTick() {
  if (!ticking) return;
  const sample = graySample();
  const threshold = Number($('sensitivity').value);
  if (lastSentFrame && meanDelta(sample, lastSentFrame) < threshold) {
    setStatus('tick skipped — scene unchanged');
    scheduleTick();
    return;
  }
  const frame = captureFrame();
  if (!frame) { scheduleTick(); return; }
  if (busy) {
    pendingFrame = { frame, sample };  // keep only the latest; flushed when free
    scheduleTick();
    return;
  }
  await sendTick(frame, sample);
  scheduleTick();
}

async function sendTick(frame, sample) {
  busy = true;
  setStatus('tick → vision + reasoning…');
  try {
    const resp = await fetch('/v2/tick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: frame }),
    });
    const data = await resp.json();
    if (data.skipped) { setStatus('tick throttled by server'); return; }
    lastSentFrame = sample;
    if (data.caption) addCaptionDot(data.caption);
    (data.triggers || []).forEach((t) => addMsg('trigger', `⚡ ${t}`));
    if (data.text) { addMsg('assistant', data.text); speak(data.text); }
    updateDoc(data.doc);
    setStatus(data.text ? 'spoke' : 'silent tick');
  } catch (e) {
    setStatus(`tick failed: ${e.message}`);
  } finally {
    busy = false;
    flushPending();
  }
}

// A queued user message always goes before a buffered frame: the person is
// waiting on an answer, the frame is only ever a few seconds of staleness.
function flushPending() {
  if (busy) return;
  if (queuedPrompt) {
    const prompt = queuedPrompt;
    queuedPrompt = null;
    deliverMessage(prompt);
    return;
  }
  if (pendingFrame && ticking) {
    const { frame, sample } = pendingFrame;
    pendingFrame = null;
    sendTick(frame, sample);
  }
}

// ── Chat ─────────────────────────────────────────────────────────────────────

async function sendMessage() {
  const input = $('chat-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  addMsg('user', prompt);
  // A tick holds `busy` for its whole vision+reasoning round trip, which on a
  // 4s interval is most of the wall clock. Dropping the message here (the
  // original behavior) meant speaking or hitting Enter mid-tick did nothing
  // at all — no reply, no error, and with voice input no visible input box to
  // notice it in. Queue it instead, exactly as pendingFrame does for frames.
  if (busy) {
    queuedPrompt = prompt;
    setStatus('queued — waiting for the current tick to finish…');
    return;
  }
  await deliverMessage(prompt);
}

async function deliverMessage(prompt) {
  busy = true;
  setStatus('thinking…');
  try {
    const body = { prompt };
    const frame = captureFrame();  // auto-attach current frame when camera is on
    if (frame) body.image_base64 = frame;
    const resp = await fetch('/v2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.caption && $('show-captions').checked) addCaptionDot(data.caption);
    addMsg('assistant', data.text || '(no reply)');
    if (data.text) speak(data.text);
    updateDoc(data.doc);
    setStatus('idle');
  } catch (e) {
    addMsg('system', `Chat failed: ${e.message}`);
    setStatus('idle');
  } finally {
    busy = false;
    flushPending();
  }
}

// ── Poll heartbeat (fired expectations while no ticks are running) ───────────

async function pollTriggers() {
  if (busy) return;
  try {
    const resp = await fetch('/v2/poll');
    const data = await resp.json();
    if (data.message) {
      (data.triggers || []).forEach((t) => addMsg('trigger', `⚡ ${t}`));
      addMsg('assistant', `⏰ ${data.message}`);
      speak(data.message);  // an expectation firing is the main thing worth hearing
    }
    updateDoc(data.doc);
  } catch (_) { /* transient — next poll will catch up */ }
}

// ── Doc panel ────────────────────────────────────────────────────────────────

async function refreshDoc() {
  try {
    const resp = await fetch('/v2/doc');
    const data = await resp.json();
    updateDoc(data.rendered);
  } catch (_) {}
}

// ── Voice output (Web Speech API — on-device, free, no server call) ──────────
// Ported from app.js. This is the hands-free payoff: a tick that decides to
// speak, or an expectation firing while your hands are in the dal, reaches you
// without looking at the screen. On by default here (unlike v1) because a live
// tick loop you can't hear is just a screen you have to watch.

const TTS_KEY = 'chitragupt-live-tts';
const synth = window.speechSynthesis || null;
let ttsEnabled = true;

function initTts() {
  const btn = $('tts-btn');
  if (!synth) { btn.style.display = 'none'; return; }
  const saved = localStorage.getItem(TTS_KEY);
  ttsEnabled = saved === null ? true : saved === '1';
  updateTtsBtn();
}

function toggleTts() {
  if (!synth) return;
  ttsEnabled = !ttsEnabled;
  try { localStorage.setItem(TTS_KEY, ttsEnabled ? '1' : '0'); } catch { /* ignore */ }
  updateTtsBtn();
  if (!ttsEnabled) synth.cancel();
}

function updateTtsBtn() {
  const btn = $('tts-btn');
  btn.classList.toggle('active', ttsEnabled);
  btn.textContent = ttsEnabled ? '🔊' : '🔇';
  btn.title = ttsEnabled ? 'Voice replies on — tap to mute' : 'Voice replies off — tap to enable';
}

function ttsCleanText(text) {
  return String(text || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[*_`#>]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// Cancel anything mid-utterance first — in a live loop the newest thing to say
// supersedes a stale one, same "latest matters" logic as pendingFrame. Letting
// narration queue up means hearing about the onions after they've burned.
function speak(text) {
  if (!ttsEnabled || !synth) return;
  const clean = ttsCleanText(text);
  if (!clean) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(clean);
  u.lang = 'en-US';
  u.rate = 1.0;
  synth.speak(u);
}

// ── Voice input (Web Speech API) ─────────────────────────────────────────────
// Chrome/Edge/Safari only (not Firefox), and needs a secure context — same
// requirement as getUserMedia, so if the camera works the mic will too. The
// button hides itself entirely rather than showing a dead control.

const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let isRecording = false;

function initVoiceInput() {
  const micBtn = $('mic-btn');
  if (!SpeechRecognitionImpl) { micBtn.style.display = 'none'; return; }

  recognizer = new SpeechRecognitionImpl();
  recognizer.lang = 'en-US';
  recognizer.continuous = false;
  recognizer.interimResults = true;

  recognizer.onresult = (event) => {
    let transcript = '';
    for (let i = 0; i < event.results.length; i++) transcript += event.results[i][0].transcript;
    $('chat-input').value = transcript;
  };

  recognizer.onerror = () => { isRecording = false; micBtn.classList.remove('active'); };

  // continuous=false, so this fires when the browser hears you stop talking.
  // Send automatically: speaking the question is the whole interaction, no
  // follow-up tap. sendMessage queues if a tick is mid-flight, so unlike the
  // old behavior a question spoken over a tick is never lost.
  recognizer.onend = () => {
    isRecording = false;
    micBtn.classList.remove('active');
    if ($('chat-input').value.trim()) sendMessage();
  };
}

function toggleVoiceInput() {
  if (!recognizer) return;
  if (isRecording) { recognizer.stop(); return; }
  // Silence any spoken reply before opening the mic, so the assistant isn't
  // talking over you — and so its own voice can't bleed into the recognizer.
  if (synth) synth.cancel();
  $('chat-input').value = '';
  isRecording = true;
  $('mic-btn').classList.add('active');
  try {
    recognizer.start();
  } catch {
    isRecording = false;
    $('mic-btn').classList.remove('active');
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────────

$('camera-btn').addEventListener('click', () => (stream ? stopCamera() : startCamera()));
$('tick-btn').addEventListener('click', () => (ticking ? stopTicks() : startTicks()));
$('interval').addEventListener('input', () => { $('interval-val').textContent = `${$('interval').value}s`; });
$('send-btn').addEventListener('click', sendMessage);
$('chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
$('doc-refresh').addEventListener('click', refreshDoc);
$('reset-btn').addEventListener('click', async () => {
  if (!confirm('Clear the world document and conversation?')) return;
  await fetch('/v2/reset', { method: 'POST' });
  $('transcript').innerHTML = '';
  refreshDoc();
  addMsg('system', 'World document cleared.');
});

$('mic-btn').addEventListener('click', toggleVoiceInput);
$('tts-btn').addEventListener('click', toggleTts);

initTts();
initVoiceInput();

setInterval(pollTriggers, POLL_INTERVAL_MS);
refreshDoc();
