"""Short-lived registry of local files the plugin is willing to serve back
to Vylen Cloud over the existing gateway tunnel.

Hermes drops generated images (and other media) onto the container's local
filesystem; we need to make those bytes reachable from the client without
either (a) inlining them on the push frame — bloats the SSE channel — or
(b) handing the cloud arbitrary read access to the container's filesystem.

The contract:
  - The adapter calls `register(path, …) -> token` when Hermes hands us a
    media path. We mint a random token, remember the (path, mime, filename,
    expiry) tuple, and return the token.
  - The token is included in the push frame the cloud relays to the client.
  - The client GETs `/v1/instances/<id>/blobs/<token>` against the cloud.
  - The cloud tunnels that request to the plugin at the magic path
    `/__vylen_blob__/<token>`; the relay intercepts that prefix, calls
    `lookup(token)`, opens the file, and streams it back via the standard
    response_chunk frames.

Because the cloud can only ask for tokens *we minted*, the registry
prevents the cloud from coercing arbitrary file reads (e.g. /etc/passwd).
Entries expire to keep the table from growing unbounded across long
Hermes sessions; the default 30-minute TTL is long enough for a user to
notice and tap an image, short enough that a forgotten image stops
existing in addressable space relatively quickly.
"""

from __future__ import annotations

import asyncio
import mimetypes
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Magic path prefix the relay watches for. Tunnel requests with paths
# starting here are served from this registry instead of forwarded to
# Hermes's HTTP API. Picked to be obviously not a real Hermes endpoint.
BLOB_PATH_PREFIX = "/__vylen_blob__/"

DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per file


@dataclass
class BlobEntry:
    path: Path
    mime: str
    filename: str
    expires_at: float  # monotonic seconds


class BlobRegistry:
    """In-memory `{token: BlobEntry}` map with TTL-based eviction.

    Thread-affinity assumption: the plugin runs as a single asyncio loop.
    All accesses come from coroutines on that loop, so an `asyncio.Lock`
    is enough — no threading primitives needed.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS):
        self._entries: dict[str, BlobEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds

    async def register(self, path: str | Path) -> Optional[tuple[str, str, str]]:
        """Add `path` to the registry. Returns `(token, mime, filename)` on
        success or `None` if the file is missing / not a regular file.

        Doesn't read the file — the bytes are only loaded when the cloud
        tunnels a request for the token. Keeps register() cheap so the
        adapter's send_image_file stays fast.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            return None
        mime, _ = mimetypes.guess_type(p.name)
        if mime is None:
            mime = "application/octet-stream"
        token = secrets.token_urlsafe(16)
        entry = BlobEntry(
            path=p,
            mime=mime,
            filename=p.name,
            expires_at=time.monotonic() + self._ttl,
        )
        async with self._lock:
            self._entries[token] = entry
            self._evict_expired_locked()
        return token, mime, p.name

    async def lookup(self, token: str) -> Optional[BlobEntry]:
        """Resolve `token` to its `BlobEntry`. Returns `None` if the token is
        unknown or has expired. Does NOT consume — repeated lookups within
        the TTL window are valid (e.g. user re-opens the chat)."""
        async with self._lock:
            self._evict_expired_locked()
            return self._entries.get(token)

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        stale = [tok for tok, e in self._entries.items() if e.expires_at <= now]
        for tok in stale:
            del self._entries[tok]
