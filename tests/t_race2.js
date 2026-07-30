// Replays the input-hijack race against the REAL app.js.
// NOTE: top-level `let` in a vm script is NOT a property of the sandbox object,
// so all state must be poked via runInContext, which resolves the script's own
// lexical bindings. Setting sandbox.liveActive silently does nothing.
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || 'd:/CV Exercise/AI_Chitragupt/server/static/app.js';

function makeEl(id) {
  return {
    id, value: '', videoWidth: 640, videoHeight: 480, width: 0, height: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    style: {}, dataset: {}, children: [], disabled: false, checked: false,
    appendChild() {}, removeChild() {}, addEventListener() {}, remove() {},
    scrollIntoView() {}, focus() {}, click() {}, insertAdjacentHTML() {},
    insertBefore() {}, firstChild: null, lastChild: null, parentNode: null,
    closest: () => null, setAttribute() {}, getAttribute: () => null,
    getContext: () => ({
      drawImage() {},
      getImageData: () => ({ data: new Uint8ClampedArray(4 * 256) }),
    }),
    toDataURL: () => 'data:image/jpeg;base64,STUBFRAME',
    querySelector: () => makeEl('q'), querySelectorAll: () => [],
    set innerHTML(v) {}, get innerHTML() { return ''; },
    set textContent(v) {}, get textContent() { return ''; },
  };
}

const els = {};
const sent = [];

const sandbox = {
  console, setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
  Date, Math, JSON, Promise, Uint8ClampedArray, URL, Object, Array, String,
  Number, Boolean, Error, TextDecoder,
  addEventListener() {}, removeEventListener() {},
  navigator: { serviceWorker: { register: () => Promise.resolve() }, mediaDevices: {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  speechSynthesis: null,
  fetch: async (url, opts) => {
    const body = JSON.parse(opts.body);
    sent.push({ url, body });
    return {
      ok: true, status: 200, body: null,
      json: async () => ({
        text: 'ok', provider: 'p', model: 'm', tool_calls: [],
        think_blocks: [], frame_detail: 'coarse',
      }),
    };
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.document = {
  getElementById: (id) => (els[id] = els[id] || makeEl(id)),
  querySelector: () => makeEl('q'), querySelectorAll: () => [],
  addEventListener() {}, createElement: () => makeEl('new'), body: makeEl('body'),
};
sandbox.self = sandbox;

vm.createContext(sandbox);
try {
  vm.runInContext(fs.readFileSync(FILE, 'utf8'), sandbox, { filename: FILE });
} catch (e) {
  console.log('load warning:', e.message);
}

const run = (code) => vm.runInContext(code, sandbox);
const input = sandbox.document.getElementById('prompt-input');
const fail = [];
const ok = (c, m) => { console.log((c ? 'PASS  ' : 'FAIL  ') + m); if (!c) fail.push(m); };

(async () => {
  // Sanity: prove we can actually reach the script's own bindings.
  run('liveActive = true; liveSending = false; lastSentDiffData = null; framesSent = 0;');
  ok(run('liveActive') === true, 'harness can set liveActive (else every test below is vacuous)');
  const hasQueue = (() => { try { run('queuedLivePrompt'); return true; } catch { return false; } })();
  console.log('      (queuedLivePrompt binding present: ' + hasQueue + ')');
  const q = () => (hasQueue ? run('queuedLivePrompt') : undefined);

  // ── 1. THE RACE: mid-typing, a tick fires ────────────────────────────────
  input.value = 'how much sal';
  sent.length = 0;
  await run('sampleLiveFrame()');
  ok(!sent.some((s) => s.body.prompt === 'how much sal'), 'tick does NOT send the half-typed question');
  ok(input.value === 'how much sal', 'tick does NOT clear the textarea');
  const tick = sent.find((s) => /Watch tick/.test(s.body.prompt || ''));
  ok(!!tick, 'tick still sends its own autonomous watch frame');
  ok(tick && tick.body.is_live_frame === true, 'autonomous frame flagged is_live_frame=true');

  // ── 2. COMMIT via Enter/Send ─────────────────────────────────────────────
  run('liveSending = false;');
  input.value = 'how much salt should I add';
  sent.length = 0;
  await run('sendMessage()');
  const c = sent.find((s) => s.body.prompt === 'how much salt should I add');
  ok(!!c, 'committed question IS sent');
  ok(input.value === '', 'commit clears the textarea');
  ok(c && c.url === '/v1/chat', 'commit goes out on the live path (/v1/chat), not frameless stream');
  ok(c && !!c.body.image_base64, 'committed question rides WITH a frame');
  ok(c && c.body.is_live_frame === false, 'committed question NOT flagged as a silent tick');

  // ── 3. no double-send ────────────────────────────────────────────────────
  run('liveSending = false;');
  sent.length = 0;
  await run('sampleLiveFrame()');
  ok(!sent.some((s) => s.body.prompt === 'how much salt should I add'), 'next tick does NOT resend it');

  // ── 4. typing again is still safe ────────────────────────────────────────
  run('liveSending = false;');
  input.value = 'and how long';
  sent.length = 0;
  await run('sampleLiveFrame()');
  ok(input.value === 'and how long', 'a second half-typed question survives a tick');

  // ── 5. commit while a request is in flight is not lost ───────────────────
  run('liveSending = true;');
  input.value = 'is it boiling';
  sent.length = 0;
  await run('sendMessage()');
  ok(q() === 'is it boiling', 'commit during an in-flight request is queued, not dropped');
  ok(input.value === '', 'textarea cleared even when queued');

  console.log('\n' + (fail.length ? fail.length + ' FAILURE(S)' : 'ALL PASS  (' + FILE.split('/').pop() + ')'));
  process.exit(fail.length ? 1 : 0);
})();
