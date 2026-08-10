// A blank camera must not look like a still scene.
//
// The diff gate compares a frame against the last one sent. A covered lens, a
// camera another app has grabbed, or a track that opened but never delivered
// all produce a FLAT frame — and a flat frame is genuinely identical to the
// last flat frame, so the gate reports "scene unchanged" truthfully and forever.
// Every tick skipped, every caption reused, the assistant silent, and nothing
// saying why. DECISIONS.md 6.2: never add a filter whose "no" looks like silence.
//
// Drives the REAL live.js in a stubbed DOM. Runs frameIsFlat against synthetic
// samples and checks onTick routes a blank frame to the fault path, not the
// gate.
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || 'd:/CV Exercise/AI_Chitragupt/server/static/live.js';
const FAIL = [];

function check(label, cond, detail = '') {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${label}` + (!cond && detail ? `   [${detail}]` : ''));
  if (!cond) FAIL.push(label);
}

const messages = [];
const flashes = [];

function makeEl(id) {
  return {
    id, value: id === 'sensitivity' ? '6' : id === 'interval' ? '4' : '',
    videoWidth: 640, videoHeight: 480, width: 0, height: 0,
    classList: {
      add(c) { if (id === 'camera-wrap') flashes.push(c); },
      remove() {}, toggle() {}, contains: () => false,
    },
    style: {}, dataset: {}, children: [], disabled: false, checked: false,
    appendChild() {}, removeChild() {}, addEventListener() {}, remove() {},
    scrollIntoView() {}, focus() {}, click() {},
    getContext: () => ({
      drawImage() {},
      getImageData: () => ({ data: sandbox.__pixels }),
    }),
    toDataURL: () => 'data:image/jpeg;base64,STUBFRAME',
    querySelector: () => makeEl('q'), querySelectorAll: () => [],
    set innerHTML(v) {}, get innerHTML() { return ''; },
    set textContent(v) {}, get textContent() { return ''; },
  };
}

const els = {};
const sandbox = {
  console, setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
  Date, Math, JSON, Promise, Float32Array, Uint8ClampedArray, URL, Object, Array,
  String, Number, Boolean, Error,
  addEventListener() {}, removeEventListener() {},
  navigator: { mediaDevices: {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  speechSynthesis: null,
  fetch: () => Promise.resolve({ json: () => Promise.resolve({}) }),
  Blob: function () {}, FormData: function () {},
  __pixels: new Uint8ClampedArray(32 * 32 * 4),
};
sandbox.window = sandbox;
sandbox.document = {
  getElementById: (id) => (els[id] = els[id] || makeEl(id)),
  createElement: (t) => makeEl(t),
  addEventListener() {}, body: makeEl('body'),
};

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(FILE, 'utf8'), sandbox, { filename: FILE });

// Replace addMsg so we can read what the user would have been told.
vm.runInContext('addMsg = (kind, text) => __msgs.push([kind, text]);', sandbox);
sandbox.__msgs = messages;

// ── Synthetic samples ────────────────────────────────────────────────────────
const flat = (v) => { const g = new Float32Array(1024); g.fill(v); return g; };
function noisy(base, spread) {
  const g = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) g[i] = base + ((i * 37) % spread);
  return g;
}

const isFlat = (g) => vm.runInContext('frameIsFlat', sandbox)(g);

console.log('\n[1] frameIsFlat — what counts as "no picture"');
check('pure black is flat', isFlat(flat(0)));
check('pure white is flat', isFlat(flat(255)), 'a lens on a white worktop is as dead as on a black one');
check('stuck mid-grey is flat', isFlat(flat(128)));
check('a real scene is NOT flat', !isFlat(noisy(40, 120)));
check('a dim but real scene is NOT flat', !isFlat(noisy(8, 40)),
      'a dark kitchen must still count as a picture');
check('sensor noise alone IS flat', isFlat(noisy(10, 2)),
      'a capped lens still jitters a little; that is not a scene');
check('a null sample is not reported as flat', !isFlat(null),
      'no sample at all is a different fault with a different path');

console.log('\n[2] a blank frame must not be reported as "unchanged"');
// Black pixels, camera "running", and a prior sent frame that was also black —
// the exact state where the gate would say "scene unchanged" forever.
vm.runInContext('stream = { getVideoTracks: () => [{ label: "back", readyState: "live", enabled: true, muted: false }] };', sandbox);
vm.runInContext('lastSentFrame = graySample(); ticking = true; flatSince = 0;', sandbox);
messages.length = 0; flashes.length = 0;
vm.runInContext('onTick();', sandbox);

const warned = messages.filter((m) => String(m[1]).includes('blank frame'));
check('the user is told the camera is blank', warned.length === 1,
      JSON.stringify(messages));
check('the warning names the track state',
      warned.length > 0 && /readyState=live/.test(warned[0][1]));
check('a blank frame flashes its own colour, not "skip"',
      flashes.includes('cap-blank') && !flashes.includes('cap-skip'),
      JSON.stringify(flashes));

messages.length = 0;
vm.runInContext('onTick();', sandbox);
check('it does not repeat the warning every tick',
      !messages.some((m) => String(m[1]).includes('blank frame')),
      JSON.stringify(messages));

console.log('\n[3] recovery is announced');
sandbox.__pixels = (() => {
  const d = new Uint8ClampedArray(32 * 32 * 4);
  for (let i = 0; i < 32 * 32; i++) { d[i * 4] = (i * 7) % 200; d[i * 4 + 1] = (i * 13) % 200; d[i * 4 + 2] = (i * 3) % 200; }
  return d;
})();
messages.length = 0;
vm.runInContext('onTick();', sandbox);
check('recovery is announced once', messages.some((m) => String(m[1]).includes('showing a picture again')),
      JSON.stringify(messages));

console.log('\n[4] caption reuse refuses to run on a blank camera');
sandbox.__pixels = new Uint8ClampedArray(32 * 32 * 4);   // black again
vm.runInContext('flatSince = 0; lastCaptionAt = Date.now(); lastSentFrame = graySample();', sandbox);
messages.length = 0;
const fresh = vm.runInContext('needsFreshCaption()', sandbox);
check('a blank camera does not silently reuse a stale caption', fresh === false);
check('and it says so rather than going quiet',
      messages.some((m) => String(m[1]).includes('blank frame')), JSON.stringify(messages));

console.log('\n' + (FAIL.length ? 'FAILURES: ' + FAIL.join(', ') : 'ALL BLANK-FRAME CHECKS PASSED'));
process.exit(FAIL.length ? 1 : 0);
