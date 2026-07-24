const CACHE_NAME = 'chitragupt-shell-v14';
const SHELL_URLS = [
  '/',
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
  if (
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
