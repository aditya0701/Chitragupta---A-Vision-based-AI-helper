// A user turn must not wait behind a tick — in the BROWSER, not just the server.
//
// The server was restructured so ticks and chat overlap: the lock covers writes
// only, and a tick abandons its own reasoning when someone is waiting. None of
// that reached the user, because live.js held a single `busy` flag covering
// both and refused to put the message on the wire until the tick returned. The
// user saw "queued — waiting for the current tick to finish" for a constraint
// that existed nowhere else in the system.
//
// Also covers the race that concurrency introduces: responses can arrive out of
// order, so a slow tick's stale doc must not overwrite a newer one.
//
// NOTE: top-level `let` in a vm script is NOT a property of the sandbox object,
// so state must be read and poked via runInContext.
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || 'd:/CV Exercise/AI_Chitragupt/server/static/live.js';
const FAIL = [];

function check(label, cond, detail = '') {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${label}` + (!cond && detail ? `   [${detail}]` : ''));
  if (!cond) FAIL.push(label);
}

const calls = [];       // every fetch that actually left the client
const resolvers = [];   // hold each response open until we choose
const rejecters = [];   // ...or fail it, for the timeout / dropped-connection path
let painted = '';       // what the doc panel currently shows

function makeEl(id) {
  const el = {
    id, value: id === 'sensitivity' ? '6' : id === 'interval' ? '4' : '',
    videoWidth: 640, videoHeight: 480, width: 0, height: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    style: {}, dataset: {}, children: [], disabled: false, checked: false,
    appendChild() {}, removeChild() {}, addEventListener() {}, remove() {},
    scrollIntoView() {}, focus() {}, click() {},
    getContext: () => ({
      drawImage() {},
      getImageData: () => ({ data: sandbox.__pixels }),
    }),
    toDataURL: () => 'data:image/jpeg;base64,STUBFRAME',
    querySelector: () => makeEl('q'), querySelectorAll: () => [],
    set innerHTML(v) { if (id === 'doc-panel') painted = v; },
    get innerHTML() { return ''; },
    set textContent(v) { if (id === 'doc-panel') painted = v; },
    get textContent() { return ''; },
  };
  return el;
}

const els = {};
// A real scene, so the diff gate and the blank-frame check both pass.
const scenePixels = (() => {
  const d = new Uint8ClampedArray(32 * 32 * 4);
  for (let i = 0; i < 32 * 32; i++) { d[i * 4] = (i * 7) % 200; d[i * 4 + 1] = (i * 13) % 200; d[i * 4 + 2] = (i * 3) % 200; }
  return d;
})();

const sandbox = {
  console, setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
  Date, Math, JSON, Promise, Float32Array, Uint8ClampedArray, URL, Object, Array,
  String, Number, Boolean, Error,
  addEventListener() {}, removeEventListener() {},
  navigator: { mediaDevices: {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  speechSynthesis: null,
  Blob: function () {}, FormData: function () {},
  // Every browser that can do getUserMedia has this, so the stub belongs here
  // rather than being defended against in live.js. Without it fetchWithTimeout
  // throws before the request leaves, and every tick assertion below fails for
  // a reason that has nothing to do with what they test.
  AbortController: function () {
    this.signal = { aborted: false };
    this.abort = () => { this.signal.aborted = true; };
  },
  __pixels: scenePixels,
  fetch: (url) => {
    calls.push(String(url));
    return new Promise((res, rej) => {
      resolvers.push((data) => res({ json: () => Promise.resolve(data) }));
      rejecters.push(rej);   // so a harness can fail a request, not just answer it
    });
  },
};
sandbox.window = sandbox;
sandbox.document = {
  getElementById: (id) => (els[id] = els[id] || makeEl(id)),
  createElement: (t) => makeEl(t),
  addEventListener() {}, body: makeEl('body'),
};

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(FILE, 'utf8'), sandbox, { filename: FILE });
vm.runInContext('addMsg = () => {}; speak = () => {};', sandbox);
vm.runInContext('stream = { getVideoTracks: () => [{ readyState: "live", enabled: true, muted: false }] };', sandbox);

const get = (expr) => vm.runInContext(expr, sandbox);
const flush = () => new Promise((r) => setImmediate(r));

(async () => {
  console.log('\n[1] a question goes out WHILE a tick is in flight');
  vm.runInContext('ticking = true; lastSentFrame = null; tickBusy = false; chatBusy = false;', sandbox);
  vm.runInContext('onTick();', sandbox);
  await flush();
  check('the tick left the client', calls.some((u) => u.includes('/v2/tick')), JSON.stringify(calls));
  check('the tick is marked in flight', get('tickBusy') === true);

  els['chat-input'].value = 'where are the onions?';
  vm.runInContext('sendMessage();', sandbox);
  await flush();

  check('the question left the client immediately, mid-tick',
        calls.some((u) => u.includes('/v2/chat')), JSON.stringify(calls));
  check('it was NOT parked in the queue', get('queuedPrompt') === null,
        JSON.stringify(get('queuedPrompt')));
  check('both are in flight at once', get('tickBusy') === true && get('chatBusy') === true);

  console.log('\n[2] a second question DOES queue behind the first');
  els['chat-input'].value = 'and the garlic?';
  vm.runInContext('sendMessage();', sandbox);
  await flush();
  check('only one chat request is on the wire',
        calls.filter((u) => u.includes('/v2/chat')).length === 1);
  check('the second question is queued, not dropped',
        get('queuedPrompt') === 'and the garlic?', JSON.stringify(get('queuedPrompt')));

  console.log('\n[3] a stale tick reply cannot overwrite a newer chat reply');
  // The chat turn (issued second) finishes FIRST and paints rev 9.
  resolvers[1]({ text: 'by the sink', doc: 'NEWER DOC', doc_rev: 9 });
  await flush();
  check('the chat doc is painted', painted === 'NEWER DOC', painted);

  // The tick (issued first) now returns carrying a doc from before that write.
  resolvers[0]({ caption: 'c', doc: 'OLDER DOC', doc_rev: 4 });
  await flush();
  check('the older render is dropped, not painted', painted === 'NEWER DOC', painted);

  console.log('\n[4] ticks still refuse to stack on each other');
  // lastSentFrame is now the scene the completed tick sent, and the pixels have
  // not moved — so the diff gate would skip this tick as "scene unchanged"
  // before it ever reached the stacking check. Clear it so the gate passes and
  // the thing under test is actually exercised.
  vm.runInContext('tickBusy = true; pendingFrame = null; lastSentFrame = null;', sandbox);
  calls.length = 0;
  vm.runInContext('onTick();', sandbox);
  await flush();
  check('no second tick went out', !calls.some((u) => u.includes('/v2/tick')), JSON.stringify(calls));
  check('the frame was buffered instead', get('pendingFrame') !== null);

  console.log('\n[5] a failed tick must not look like a silent one');
  // /v2/tick answers 200 with an `error` field, so the fetch catch never fires.
  // A backend refusal (DECISIONS.md 5.3) returns this on EVERY frame — if it
  // falls through to 'silent tick' the user watches a dead system behave
  // exactly like a working, quiet one.
  const msgs = [];
  vm.runInContext('addMsg = (kind, text) => { __msgs.push(kind + "|" + text); };', sandbox);
  sandbox.__msgs = msgs;
  // pendingFrame MUST be cleared: section [4] deliberately left one buffered,
  // and the finally-block flush would re-enter onTick and overwrite the status
  // line before it could be read — making the status assertion below pass for
  // a reason that has nothing to do with what it claims to test.
  vm.runInContext(
    'tickBusy = false; pendingFrame = null; lastSentFrame = null; lastTickError = null;',
    sandbox);
  calls.length = 0; resolvers.length = 0;
  vm.runInContext('onTick();', sandbox);
  await flush();
  resolvers[0]({ text: null, error: 'v2 refuses to start with vision on Groq' });
  await flush();
  check('the failure is reported to the user',
        msgs.some((m) => m.includes('refuses to start')), JSON.stringify(msgs));
  // Read the status LINE, not the message list — 'silent tick' is written by
  // setStatus, so asserting against addMsg output would pass vacuously and
  // prove nothing about the failure this section exists for.
  check('the status line does not claim a silent tick',
        !/silent/i.test(get('tickStatus')), get('tickStatus'));

  // Every frame carries the same refusal; one line per tick would bury the log.
  const seen = msgs.length;
  vm.runInContext('tickBusy = false; lastSentFrame = null;', sandbox);
  vm.runInContext('onTick();', sandbox);
  await flush();
  resolvers[1]({ text: null, error: 'v2 refuses to start with vision on Groq' });
  await flush();
  check('an identical repeat is not re-announced', msgs.length === seen,
        JSON.stringify(msgs.slice(seen)));

  console.log('\n[6] a tick that never comes back is announced, not just status-lined');
  // The other way a tick ends without writing anything. An aborted deadline, a
  // dropped connection and a dyno that restarted mid-request (its 502 is HTML,
  // so resp.json() throws) all land in the same catch. On 2026-09-06 this path
  // ran for minutes while the transcript stayed empty and the status line held
  // "tick → vision + reasoning…", which reads exactly like a tick still working.
  msgs.length = 0;
  vm.runInContext('tickBusy = false; pendingFrame = null; lastSentFrame = null; lastTickError = null;',
                  sandbox);
  calls.length = 0; resolvers.length = 0; rejecters.length = 0;
  vm.runInContext('onTick();', sandbox);
  await flush();
  rejecters[0](Object.assign(new Error('signal is aborted without reason'),
                             { name: 'AbortError' }));
  await flush();
  check('the timeout is reported to the user',
        msgs.some((m) => m.includes('did not complete')), JSON.stringify(msgs));
  // The raw AbortError message names no cause and no duration. The user needs
  // to know it gave up, after how long, and that the loop is still alive.
  check('it says how long it waited, not "signal is aborted"',
        msgs.some((m) => m.includes('45s') && !m.includes('without reason')),
        JSON.stringify(msgs));
  check('the status line does not claim a silent tick',
        !/silent/i.test(get('tickStatus')), get('tickStatus'));

  console.log('\n[7] a wiped world document is announced, not read as a stale render');
  // worlddoc.clear() unlinks the file, so the server restarts at rev 0 — and on
  // Render's free plan (no persistent disk) a restart does the same thing with
  // nobody asking. The out-of-order gate would read that as a late response and
  // freeze the panel on a document that no longer exists, for as many ticks as
  // the server needs to climb back. Confirmed live: rev 14 -> 6 mid-session.
  msgs.length = 0;
  painted = '';
  vm.runInContext('lastDocRev = 14;', sandbox);
  vm.runInContext('updateDoc("FRESH EMPTY DOC", 1);', sandbox);
  check('the fresh document is painted', painted === 'FRESH EMPTY DOC', painted);
  check('the user is told the session was lost',
        msgs.some((m) => m.includes('reset')), JSON.stringify(msgs));
  check('the client re-syncs to the new revision', get('lastDocRev') === 1,
        String(get('lastDocRev')));

  // The slack is what keeps [3] working: overlapping turns arrive a few revs
  // out of order and must still be dropped, not mistaken for a wipe.
  msgs.length = 0;
  painted = 'CURRENT DOC';
  vm.runInContext('lastDocRev = 14;', sandbox);
  vm.runInContext('updateDoc("SLIGHTLY OLDER DOC", 11);', sandbox);
  check('a small backwards step is still just a stale render',
        painted === 'CURRENT DOC' && msgs.length === 0,
        painted + ' ' + JSON.stringify(msgs));

  console.log('\n' + (FAIL.length ? 'FAILURES: ' + FAIL.join(', ') : 'ALL CONCURRENCY CHECKS PASSED'));
  process.exit(FAIL.length ? 1 : 0);
})();
