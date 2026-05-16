"""In-memory buffer for in-flight Hermes responses, keyed by response_id.

Enables resumable streaming: when a mobile client's SSE socket drops
(screen off, network change), it reconnects with the response_id and
last cursor and replays missed deltas from this buffer. See
docs/specs/003-resumable-responses.md.

Buffers live only on the user's local plugin process — never on Vylen
Cloud. Plugin restart wipes the registry; the user's fallback is asking
Hermes to resend (which has its own conversation log).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_GRACE_SECONDS = 300.0
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
_PARSER_BUFFER_SOFT_CAP = 1 << 16  # 64 KiB; if we haven't seen response.created by here it likely never comes


@dataclass
class ResponseBuffer:
    """Per-response chunk store. `chunks` holds raw bytes in arrival order;
    `progressed` is set on every append and on finalize so live tailers
    wake up to drain new content."""

    response_id: str
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=list)
    total_bytes: int = 0
    complete: bool = False
    ended_at: Optional[float] = None
    progressed: asyncio.Event = field(default_factory=asyncio.Event)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.chunks.append(chunk)
        self.total_bytes += len(chunk)
        self.progressed.set()

    def finalize(self) -> None:
        self.complete = True
        self.ended_at = time.monotonic()
        self.progressed.set()

    @property
    def cursor(self) -> int:
        return len(self.chunks)

    def slice_from(self, after_cursor: int) -> list[bytes]:
        if after_cursor < 0:
            after_cursor = 0
        return list(self.chunks[after_cursor:])


class ResponseBufferRegistry:
    """Process-local registry of `ResponseBuffer` instances. Single-loop
    asyncio access only — no cross-thread safety because the relay and
    the resume handler both run inside one event loop."""

    def __init__(
        self,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._buffers: dict[str, ResponseBuffer] = {}
        self._grace_seconds = grace_seconds
        self._max_bytes = max_bytes

    @property
    def grace_seconds(self) -> float:
        return self._grace_seconds

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def create(
        self,
        response_id: str,
        status: int,
        headers: dict[str, str],
    ) -> ResponseBuffer:
        """Replaces any existing entry with the same id (Hermes never reuses
        ids, but defensive)."""
        buf = ResponseBuffer(
            response_id=response_id,
            status=status,
            headers=dict(headers),
        )
        self._buffers[response_id] = buf
        return buf

    def get(self, response_id: str) -> Optional[ResponseBuffer]:
        return self._buffers.get(response_id)

    def drop(self, response_id: str) -> None:
        self._buffers.pop(response_id, None)

    def __len__(self) -> int:
        return len(self._buffers)

    def sweep(self, now: Optional[float] = None) -> int:
        """Evict completed buffers past grace TTL, or any buffer past the
        byte cap. Returns the number evicted. Safe to call any time;
        callers usually drive this from a background task."""
        if now is None:
            now = time.monotonic()
        to_drop: list[str] = []
        for rid, buf in self._buffers.items():
            if buf.total_bytes > self._max_bytes:
                to_drop.append(rid)
                continue
            if buf.complete and buf.ended_at is not None and (now - buf.ended_at) > self._grace_seconds:
                to_drop.append(rid)
        for rid in to_drop:
            self._buffers.pop(rid, None)
        return len(to_drop)


class ResponseIdExtractor:
    """Scans an SSE byte stream for the first `response.created` event and
    captures the Hermes response_id. Stateful across chunked feeds.

    The extractor is deliberately permissive: if Hermes changes the event
    shape, we silently never find an id and the response is forwarded
    normally but not resumable. Soft cap on internal buffer prevents
    runaway memory if the stream never produces a `response.created`."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._found: Optional[str] = None
        self._done = False

    @property
    def response_id(self) -> Optional[str]:
        return self._found

    def feed(self, chunk: bytes) -> Optional[str]:
        if self._done or not chunk:
            return None
        self._buffer.extend(chunk)
        while True:
            idx = self._buffer.find(b"\n\n")
            if idx < 0:
                if len(self._buffer) > _PARSER_BUFFER_SOFT_CAP:
                    # Keep the tail in case the event straddles the cap.
                    self._buffer = bytearray(self._buffer[-(_PARSER_BUFFER_SOFT_CAP // 2):])
                return None
            event_bytes = bytes(self._buffer[:idx])
            del self._buffer[: idx + 2]
            rid = _try_extract_response_id(event_bytes)
            if rid:
                self._found = rid
                self._done = True
                # Free the buffer; we won't parse more.
                self._buffer = bytearray()
                return rid


def _try_extract_response_id(event_bytes: bytes) -> Optional[str]:
    event_type: Optional[str] = None
    data_lines: list[bytes] = []
    for line in event_bytes.split(b"\n"):
        if line.startswith(b"event:"):
            event_type = line[len(b"event:"):].strip().decode("utf-8", errors="replace")
        elif line.startswith(b"data:"):
            data_lines.append(line[len(b"data:"):].strip())
    if event_type != "response.created" or not data_lines:
        return None
    data = b"\n".join(data_lines)
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    # Hermes emits `{"response": {"id": "resp_..."}}` in response.created;
    # accept a flat `id` too for forward-compat.
    response = obj.get("response")
    if isinstance(response, dict):
        rid = response.get("id")
        if isinstance(rid, str) and rid:
            return rid
    rid = obj.get("id")
    if isinstance(rid, str) and rid:
        return rid
    return None
