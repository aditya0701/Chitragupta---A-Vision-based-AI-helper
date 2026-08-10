// Chitragupta Live (v2) — client for the world-doc tick system.
// Independent of app.js; talks only to /v2/*.

// Capture size per detail tier. Image cost scales with RESOLUTION, not JPEG
// quality — quality is not a lever (DECISIONS.md 1.2). Coarse is ~390 image
// tokens, fine ~735, so the tier is roughly a 30% saving on every tick that
// isn't actively trying to read something.
const FRAME_DIM = { coarse: 640, fine: 1024 };
const JPEG_QUALITY = 0.85;
const POLL_INTERVAL_MS = 20000;

let stream = null;
let ticking = false;
let tickTimer = null;
let busy = false;            // a /v2 request is in flight
let pendingFrame = null;     // latest frame captured while busy — flushed when free
let queuedPrompt = null;     // user message spoken/typed while busy — flushed when free
let lastSentFrame = null;    // grayscale sample of the last frame actually sent (diff gate)
let lastCaptionAt = 0;       // when a caption last came back — how stale reuse would be
// What the server told us the NEXT tick capture should be. It works a frame
// ahead because resolution thrown away here can never be recovered server-side.
let frameDetail = 'coarse';

const $ = (id) => document.getElementById(id);
const video = $('camera-video');

// ── Session log ──────────────────────────────────────────────────────────────
// Everything that happened, in order, for the export. Kept separate from the
// DOM because the transcript hides captions behind dots and drops nothing here:
// the point of a saved session is judging a real run afterwards, and per
// DECISIONS.md's testing notes the wire log is the highest-value artifact
// there is. Captions, tool calls and tier changes all matter and none of them
// are legible on screen.
const sessionLog = [];
const logEvent = (kind, text, extra) =>
  sessionLog.push({ t: new Date().toISOString(), kind, text, ...(extra || {}) });

function exportSession() {
  const pad = (n) => String(n).padStart(2, '0');
  const d = new Date();
  const out = [
    '# Chitragupta Live (v2) session',
    `Exported ${d.toISOString()}`,
    `Entries: ${sessionLog.length}`,
    '',
    '## Transcript',
    '',
  ];
  for (const e of sessionLog) {
    const time = e.t.slice(11, 19);
    if (e.kind === 'caption') {
      out.push(`- \`${time}\` 👁 **camera** _(${e.detail || '?'})_: ${e.text}`);
    } else if (e.kind === 'tools') {
      out.push(`- \`${time}\` 🔧 tools: ${e.text}`);
    } else if (e.kind === 'urgent') {
      out.push(`- \`${time}\` ⚠️ **URGENT**: ${e.text}`);
    } else if (e.kind === 'tier') {
      out.push(`- \`${time}\` 🔍 capture tier → **${e.text}**`);
    } else {
      out.push(`- \`${time}\` **${e.kind}**: ${e.text}`);
    }
  }
  out.push('', '## World document at export', '', '```', $('doc-panel').textContent, '```');

  const blob = new Blob([out.join('\n')], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `chitragupt-live-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`
             + `-${pad(d.getHours())}${pad(d.getMinutes())}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
  setStatus(`saved ${sessionLog.length} entries`);
}

// ── Transcript ───────────────────────────────────────────────────────────────

function addMsg(kind, text) {
  logEvent(kind, text);
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

// Every caption arrives through here — tick, yielded tick, or user turn — so
// this is the one place that knows how fresh our picture of the world is.
function addCaptionDot(caption) {
  lastCaptionAt = Date.now();
  logEvent('caption', caption, { detail: frameDetail });
  const div = document.createElement('div');
  div.className = $('show-captions').checked ? 'msg system' : 'caption-dot';
  div.textContent = $('show-captions').checked ? `👁 ${caption}` : '·';
  div.title = caption;
  $('transcript').appendChild(div);
  $('transcript').scrollTop = $('transcript').scrollHeight;
}

function setStatus(text) { $('status-line').textContent = text; }

// ── Capture feedback ─────────────────────────────────────────────────────────
// Two separate signals, deliberately: the badge is STATE (what the next capture
// will be), the border flash is an EVENT (a frame just left, or didn't). The
// tick loop is otherwise completely invisible — you cannot tell a working gate
// from a broken camera, or a coarse tick from a fine one, without reading logs.

let flashTimer = null;
const CAP_CLASSES = ['cap-coarse', 'cap-fine', 'cap-user', 'cap-skip'];

function flashCapture(kind) {
  const wrap = $('camera-wrap');
  wrap.classList.remove(...CAP_CLASSES);
  // Force a reflow so back-to-back flashes of the same kind still re-trigger
  // the transition instead of looking like one long hold.
  void wrap.offsetWidth;
  wrap.classList.add(`cap-${kind}`);
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => wrap.classList.remove(...CAP_CLASSES),
                          kind === 'skip' ? 220 : 450);
}

// Shows the dimensions frames are ACTUALLY captured at, derived the same way
// captureFrame derives them. A bare "640" hid the shape of the frame entirely,
// which is how 640x360 went unnoticed.
function captureDims(detail) {
  const dim = FRAME_DIM[detail] || FRAME_DIM.coarse;
  if (!video.videoWidth) return `${dim}`;
  const scale = Math.min(1, dim / Math.max(video.videoWidth, video.videoHeight));
  return `${Math.round(video.videoWidth * scale)}×${Math.round(video.videoHeight * scale)}`;
}

function updateTierBadge() {
  const badge = $('tier-badge');
  const fine = frameDetail === 'fine';
  badge.textContent = `${captureDims(frameDetail)} · ${fine ? 'fine 🔍' : 'coarse'}`;
  badge.classList.toggle('fine', fine);
  badge.title = fine
    ? 'Looking closely — full-resolution frames so small detail and label text are readable'
    : 'Normal watching — smaller frames, ~28% cheaper per tick';
}

const escHtml = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// A plan waiting on the user's yes/no is the one thing in the document that is
// blocked ON THEM. Everything else is a record; this is a question. It gets
// highlighted so a glance at the panel answers "is it waiting for me?" — the
// spoken ask is still the primary channel, this is the backstop for when the
// kitchen was too loud or the phone was face-down.
function updateDoc(rendered) {
  if (rendered == null) return;
  const panel = $('doc-panel');
  const text = rendered || '(empty)';
  const start = text.indexOf('[PROPOSED PLAN');
  if (start < 0) { panel.textContent = text; return; }   // textContent: no escaping needed
  let end = text.indexOf('\n\n[', start);
  if (end < 0) end = text.length;
  panel.innerHTML = escHtml(text.slice(0, start))
    + '<mark class="pending-plan">' + escHtml(text.slice(start, end)) + '</mark>'
    + escHtml(text.slice(end));
}

// ── Camera ───────────────────────────────────────────────────────────────────

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment',
        // 4:3, not the 16:9 a phone gives you by default. Two reasons, and the
        // second is the real one:
        //
        // 1. FRAME_DIM caps the LONGEST side, so 16:9 at coarse is 640x360 —
        //    only 360px of vertical scene for a camera pointed down at a work
        //    surface, where vertical is where the hands and the work are.
        // 2. On most phone sensors 4:3 IS the native readout and 16:9 is a
        //    vertical crop of it. Asking for 4:3 therefore gains real field of
        //    view rather than trading it away — 640x480 sees more of the
        //    counter than 640x360 does, for ~30% more image tokens.
        //
        // All `ideal`, never `exact`: a device that cannot do 4:3 should give
        // its best match rather than failing to open the camera at all.
        width: { ideal: 1280 },
        height: { ideal: 960 },
        aspectRatio: { ideal: 4 / 3 },
      },
      audio: false,
    });
  } catch (e) {
    addMsg('system', `Camera failed: ${e.message}`);
    return;
  }
  // Listener BEFORE srcObject: loadedmetadata can fire as soon as the stream is
  // attached, and a `once` handler registered afterwards can miss it outright —
  // which is silent, because its only job is a diagnostic message.
  //
  // Report what the device actually negotiated, not what we asked for — the
  // constraints above are all `ideal`, so a phone is free to hand back 16:9
  // anyway and there is otherwise no way to tell that it did.
  video.addEventListener('loadedmetadata', () => {
    const w = video.videoWidth, h = video.videoHeight;
    // Compare on the LONG side over the short side, so a phone held upright
    // isn't misreported. A portrait 960x1280 is a 4:3 sensor readout rotated,
    // not some odd shape — the first version printed it as "0.75:1", which
    // looked like the aspect request had failed when it had actually worked.
    const long = Math.max(w, h), short = Math.min(w, h);
    const r = long / short;
    const portrait = h > w ? ' portrait' : '';
    const shape = Math.abs(r - 4 / 3) < 0.05 ? `4:3${portrait}`
                : Math.abs(r - 16 / 9) < 0.05 ? `16:9${portrait} (asked for 4:3)`
                : `${r.toFixed(2)}:1${portrait}`;
    addMsg('system', `Camera: ${w}×${h} — ${shape}, capturing ${captureDims(frameDetail)}`);
    updateTierBadge();
  }, { once: true });

  video.srcObject = stream;
  video.style.display = 'block';
  $('camera-off').style.display = 'none';
  $('camera-btn').textContent = '🎥 Stop camera';
  $('camera-btn').classList.add('active');
  $('tick-btn').disabled = false;

  // The element carries `autoplay`, but autoplay is refused often enough — a
  // backgrounded tab, a browser that wants a gesture it didn't see — and when
  // it is refused the element sits at its poster frame, which is BLACK. That
  // looks exactly like a covered lens. play() is idempotent when it is already
  // playing, so calling it costs nothing and removes one whole explanation.
  try {
    await video.play();
  } catch (e) {
    addMsg('system', `⚠️ Video playback was blocked (${e.name}) — tap the preview to start it.`);
    video.addEventListener('click', () => video.play().catch(() => {}), { once: true });
  }

  // First frames out of a camera are routinely blank while it meters and
  // focuses; that is normal and must not be reported as a fault. Only a camera
  // still blank a couple of seconds in has actually failed.
  setTimeout(() => { if (stream && frameIsFlat(graySample())) reportFlatFrame(); }, 2500);
}

function stopCamera() {
  stopTicks();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  flatSince = 0;   // a stopped camera is not a faulty one; don't warn on restart
  video.srcObject = null;
  video.style.display = 'none';
  $('camera-off').style.display = 'block';
  $('camera-btn').textContent = '🎥 Start camera';
  $('camera-btn').classList.remove('active');
  $('tick-btn').disabled = true;
}

// `detail` is explicit at every call site rather than read from the global:
// ticks follow the server's echo, but a user turn always gets a full-resolution
// frame — you asked, you get the good look — same rule as v1.
function captureFrame(detail) {
  if (!stream || video.videoWidth === 0) return null;
  const dim = FRAME_DIM[detail] || FRAME_DIM.coarse;
  const scale = Math.min(1, dim / Math.max(video.videoWidth, video.videoHeight));
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

// Is this frame a picture of anything at all?
//
// A covered lens, a camera another app has taken, a track that opened but never
// delivered — all produce a FLAT frame, and a flat frame is byte-identical to
// the last flat frame. So the diff gate reports "scene unchanged" and keeps
// reporting it forever: every tick skipped, every caption reused, the assistant
// silent, and nothing anywhere saying why. That is exactly the failure
// DECISIONS.md 6.2 warns about — a filter whose "no" is indistinguishable from
// having nothing to say.
//
// Standard deviation, not mean: a lens against a white worktop is as dead as
// one against a black one, and both are as dead as a track stuck on grey. What
// says "this is a photograph" is variation, not brightness.
const FLAT_FRAME_STDDEV = 2.0;

function frameIsFlat(gray) {
  if (!gray) return false;                 // no sample at all is a different fault
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const mean = sum / gray.length;
  let variance = 0;
  for (let i = 0; i < gray.length; i++) variance += (gray[i] - mean) ** 2;
  return Math.sqrt(variance / gray.length) < FLAT_FRAME_STDDEV;
}

// Warn once per episode, not once per tick — at a 4s interval a stuck camera
// would otherwise write fifteen identical lines a minute into the transcript.
let flatSince = 0;

function reportFlatFrame() {
  if (flatSince) return false;
  flatSince = Date.now();
  const track = stream && stream.getVideoTracks()[0];
  const state = track
    ? `track "${track.label || 'unnamed'}" readyState=${track.readyState} enabled=${track.enabled} muted=${track.muted}`
    : 'no video track';
  addMsg('system', `⚠️ Camera is delivering a blank frame — nothing to look at. `
    + `Check the lens isn't covered and that no other app has the camera. (${state}, `
    + `video ${video.videoWidth}×${video.videoHeight})`);
  logEvent('camera-blank', state);
  setStatus('camera blank — ticks paused');
  return true;
}

function clearFlatFrame() {
  if (!flatSince) return;
  addMsg('system', `✅ Camera is showing a picture again (blank for `
    + `${Math.round((Date.now() - flatSince) / 1000)}s).`);
  flatSince = 0;
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
  // Only claim idle if nothing is actually in flight. Stopping ticks or the
  // camera used to overwrite "thinking…" with "idle" while a chat request was
  // still running, which reads as though the turn was cancelled — it never
  // was, and the reply still arrives. Toggling a control must not narrate
  // someone else's request.
  if (!busy) setStatus('idle');
}

function scheduleTick() {
  if (!ticking) return;
  clearTimeout(tickTimer);
  tickTimer = setTimeout(onTick, Number($('interval').value) * 1000);
}

async function onTick() {
  if (!ticking) return;
  const sample = graySample();

  // Checked BEFORE the diff gate, because a blank frame passes the gate — it
  // is genuinely identical to the last blank one — and would be reported as a
  // quiet kitchen rather than a camera fault.
  if (frameIsFlat(sample)) {
    reportFlatFrame();
    flashCapture('blank');
    scheduleTick();
    return;
  }
  clearFlatFrame();

  const threshold = Number($('sensitivity').value);
  if (lastSentFrame && meanDelta(sample, lastSentFrame) < threshold) {
    flashCapture('skip');   // the gate working IS the main cost control — show it
    setStatus('tick skipped — scene unchanged');
    scheduleTick();
    return;
  }
  const frame = captureFrame(frameDetail);
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
  flashCapture(frameDetail === 'fine' ? 'fine' : 'coarse');
  setStatus(`tick → vision + reasoning… (${FRAME_DIM[frameDetail] || FRAME_DIM.coarse}px)`);
  try {
    const resp = await fetch('/v2/tick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: frame }),
    });
    const data = await resp.json();
    if (data.skipped) { setStatus('tick throttled by server'); return; }
    // The server dropped this tick's reasoning because a user turn was
    // waiting. Keep the caption, say nothing — it is not a silent tick.
    if (data.yielded) {
      if (data.caption) addCaptionDot(data.caption);
      updateDoc(data.doc);
      setStatus('tick yielded — answering you first');
      return;
    }
    lastSentFrame = sample;
    if (data.frame_detail && data.frame_detail !== frameDetail) {
      frameDetail = data.frame_detail; updateTierBadge(); logEvent('tier', frameDetail);
    }
    if ((data.tool_calls || []).length) logEvent('tools', data.tool_calls.map((t) => t.tool).join(', '));
    if (data.caption) addCaptionDot(data.caption);
    (data.triggers || []).forEach((t) => addMsg('trigger', `⚡ ${t}`));
    if (data.text) {
      addMsg(data.urgent ? 'urgent' : 'assistant', data.urgent ? `⚠️ ${data.text}` : data.text);
      // Urgent speech jumps any queued narration outright — a warning that
      // arrives after the sentence it interrupted is a warning that arrived late.
      if (data.urgent && synth) synth.cancel();
      speak(data.text);
    }
    updateDoc(data.doc);
    const lens = frameDetail === 'fine' ? ' · 🔍 looking closely' : '';
    setStatus((data.urgent ? '⚠️ warned' : data.text ? 'spoke' : 'silent tick') + lens);
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

// How stale a reused caption may be, however still the scene looks. The diff
// gate samples 32x32, so it cannot see a label turned over or a dial nudged —
// this is the backstop for the small change it misses.
const CAPTION_REUSE_MS = 15000;

// Should this turn pay for a fresh caption, or is the last one still true?
//
// A question used to cost a full 1024px vision round trip before DeepSeek was
// even called — 2-5s of latency on top of the answer, on every single turn,
// including "how long does dal take?" asked while standing perfectly still.
//
// The perceptual gate already answers this. If the frame is within threshold
// of the last one we sent, the caption we got back for that frame still
// describes what the camera sees; re-describing it buys a differently-worded
// paragraph and nothing else. The doc carries that caption into the prompt
// under [Recent observations] with its timestamp, so the model knows both what
// is there and how old the knowledge is.
//
// When the scene HAS moved we still capture at fine — you asked, you get the
// good look. The two cases are coherent: either the picture is the same and
// looking again is free of information, or it changed and deserves the best
// frame available.
function needsFreshCaption() {
  if (!stream) return false;
  const sample = graySample();
  // A blank frame is "unchanged" from the last blank frame, so without this the
  // reuse path would answer every question from a stale caption while the
  // camera showed nothing — and never say so.
  if (frameIsFlat(sample)) { reportFlatFrame(); return false; }
  if (!lastSentFrame || !lastCaptionAt) return true;      // nothing to reuse
  if (Date.now() - lastCaptionAt > CAPTION_REUSE_MS) return true;
  return meanDelta(sample, lastSentFrame) >= Number($('sensitivity').value);
}

async function deliverMessage(prompt) {
  busy = true;
  setStatus('thinking…');
  try {
    const body = { prompt };
    if (needsFreshCaption()) {
      const sample = graySample();
      const frame = captureFrame('fine');
      if (frame) {
        body.image_base64 = frame;
        lastSentFrame = sample;   // this frame IS now the reference for the gate
        flashCapture('user');
        logEvent('capture', 'user turn — scene changed, fresh 1024px frame');
      }
    } else if (stream) {
      flashCapture('skip');
      logEvent('capture', 'user turn — scene unchanged, reusing the last caption');
    }
    const resp = await fetch('/v2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.frame_detail && data.frame_detail !== frameDetail) {
      frameDetail = data.frame_detail; updateTierBadge(); logEvent('tier', frameDetail);
    }
    if ((data.tool_calls || []).length) logEvent('tools', data.tool_calls.map((t) => t.tool).join(', '));
    if (data.caption) addCaptionDot(data.caption);
    addMsg('assistant', data.text || '(no reply)');
    if (data.text) speak(data.text);
    updateDoc(data.doc);
    setStatus(frameDetail === 'fine' ? 'idle · 🔍 looking closely' : 'idle');
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
$('save-btn').addEventListener('click', exportSession);
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
updateTierBadge();

setInterval(pollTriggers, POLL_INTERVAL_MS);
refreshDoc();
