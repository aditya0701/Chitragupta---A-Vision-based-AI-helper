// v20: `/` now serves v2's live UI, so it left the shell. Every browser that
// had ever loaded v1 held v1's index.html cached under `/` cache-first, and
// without this bump they would keep being served it through any deploy.
const CACHE_NAME = 'chitragupt-shell-v20';
const SHELL_URLS = [
  '/v1',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls — chat/vision responses must always be fresh.
  // /v2 + /live (the parallel live tick system) are excluded from the SW
  // entirely, page and assets included, so iterating on it never fights
  // the shell cache.
  //
  // `/` is on that list because it IS the live UI now. v1's shell lives at
  // `/v1`, which is still cached, so the PWA keeps working offline — it just
  // is not what the bare origin resolves to any more.
  if (
    url.pathname === '/' ||
    url.pathname.startsWith('/v1/') ||
    url.pathname.startsWith('/v2/') ||
    url.pathname === '/health' ||
    url.pathname === '/live' ||
    url.pathname.startsWith('/static/live')
  ) {
    return;
  }

  // Cache-first for the app shell (static assets).
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((resp) => {
        if (resp.ok && event.request.method === 'GET') {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return resp;
      });
    })
  );
});
