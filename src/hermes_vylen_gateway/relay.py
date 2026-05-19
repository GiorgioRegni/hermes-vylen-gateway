"""Gateway request relay: bridges gateway WS request frames to in-process
Hermes agent execution, streaming response bytes back as WS response frames.

This is the chat path. Cron push and multimodal-via-MessageEvent are a
different code path (checkpoint 6) that uses `BasePlatformAdapter.send` style
callbacks rather than HTTP-tunneling.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from . import agent_runner
from .blobs import BLOB_PATH_PREFIX, BlobRegistry
from .response_buffer import (
    ResponseBuffer,
    ResponseBufferRegistry,
    ResponseIdExtractor,
)

logger = logging.getLogger(__name__)

# Streamed-blob read chunk size. Picked to match a reasonable WebSocket
# message budget (a too-big chunk inflates the websocket buffer; a too-small
# chunk fragments the file across many frames and slows transfer). 256 KiB
# is a comfortable middle for typical Hermes-generated images.
_BLOB_CHUNK_BYTES = 256 * 1024

# Frame type constants — keep in sync with cloud/internal/cloud/gateway.go.
FRAME_REQUEST = "request"
FRAME_RESPONSE_HEADERS = "response_headers"
FRAME_RESPONSE_CHUNK = "response_chunk"
FRAME_RESPONSE_END = "response_end"
FRAME_RESPONSE_ERROR = "response_error"
FRAME_RESPONSE_RESUME = "response_resume"

def _resolve_request_timeout() -> float | None:
    """Read `VYLEN_HERMES_TIMEOUT` (seconds). Empty/unset → 5 minutes;
    "none" disables the bound (diagnostic use only). Used by HermesRelay
    so a hung local Hermes can't accumulate stuck background tasks."""
    raw = (os.environ.get("VYLEN_HERMES_TIMEOUT") or "").strip().lower()
    if raw == "none":
        return None
    if not raw:
        return 300.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("VYLEN_HERMES_TIMEOUT=%r is not a number; using default", raw)
        return 300.0


class HermesRelay:
    """Routes a single inbound `request` frame to local Hermes and streams the
    response back. Each request runs in its own asyncio task; the relay is
    owned by the adapter for the lifetime of the gateway socket.
    """

    def __init__(
        self,
        send_frame: Callable[[dict[str, Any]], Awaitable[None]],
        request_timeout: float | None = None,
        blobs: Optional[BlobRegistry] = None,
        response_buffers: Optional[ResponseBufferRegistry] = None,
    ):
        self._send = send_frame
        # Shared blob registry — populated by the adapter when Hermes
        # produces an image; served by `_serve_blob` when the cloud tunnels
        # a request to `/__vylen_blob__/<token>`.
        self._blobs = blobs
        # Response buffer registry. Optional so callers that don't need
        # resume (tests, blob-only mocks) can omit it; the adapter wires a
        # real registry in production. See spec 003.
        self._response_buffers = response_buffers
        # Bound the in-process Hermes call lifetime so a hung model/tool
        # path can't accumulate background tasks indefinitely. The default
        # mirrors the former HTTP relay timeout.
        if request_timeout is None:
            request_timeout = _resolve_request_timeout()
        self._request_timeout = request_timeout
        self._tasks: set[asyncio.Task] = set()

    async def handle(self, frame: dict[str, Any]) -> None:
        """Schedule one request. Returns immediately; reply happens in the
        background so the read loop stays free for the next frame."""
        request_id = frame.get("request_id") or ""
        if not request_id:
            logger.warning("relay: request frame missing request_id, dropping")
            return
        task = asyncio.create_task(self._run(request_id, frame))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def handle_resume(self, frame: dict[str, Any]) -> None:
        """Schedule one resume. Looks up the buffer by response_id and
        replays chunks past `after_cursor`, then tails live until the
        buffer is complete. See docs/specs/003-resumable-responses.md."""
        request_id = frame.get("request_id") or ""
        if not request_id:
            logger.warning("relay: response_resume frame missing request_id, dropping")
            return
        response_id = frame.get("response_id") or ""
        try:
            after_cursor = int(frame.get("after_cursor") or 0)
        except (TypeError, ValueError):
            after_cursor = 0
        task = asyncio.create_task(
            self._run_resume(request_id, response_id, after_cursor)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        return None

    async def _run(self, request_id: str, frame: dict[str, Any]) -> None:
        method = frame.get("method") or "GET"
        path = frame.get("path") or "/"
        body_b64 = frame.get("body") or ""
        headers = frame.get("headers") or {}
        # Blob fetch short-circuit: tunnel requests against the magic
        # `/__vylen_blob__/<token>` prefix are served from the adapter's
        # BlobRegistry, not forwarded to Hermes. The cloud only routes
        # requests with tokens we minted (see blobs.py), so this can't
        # become an arbitrary-file-read primitive for the cloud.
        if self._blobs is not None and path.startswith(BLOB_PATH_PREFIX):
            await self._serve_blob(request_id, method, path)
            return
        try:
            body = base64.b64decode(body_b64) if body_b64 else b""
        except Exception as exc:  # noqa: BLE001
            await self._send_error(request_id, "BAD_REQUEST", f"could not decode body: {exc}")
            return

        forwarded = _filter_request_headers(headers)
        logger.debug("relay -> %s %s (request_id=%s)", method, path, request_id)
        writer = _FrameStreamWriter(
            send=self._send,
            request_id=request_id,
            response_buffers=self._response_buffers,
        )
        try:
            coro = agent_runner.dispatch(method, path, forwarded, body, writer)
            if self._request_timeout is None:
                await coro
            else:
                await _await_with_progress_timeout(
                    coro,
                    writer.progress,
                    timeout=self._request_timeout,
                )
        except asyncio.TimeoutError:
            writer.abandon()
            logger.warning("relay timeout for %s %s", method, path)
            await self._send_error(request_id, "HERMES_TIMEOUT", "Hermes request timed out")
        except asyncio.CancelledError:
            writer.abandon()
            raise
        except Exception as exc:  # noqa: BLE001
            writer.abandon()
            logger.exception("relay crashed for %s %s", method, path)
            await self._send_error(request_id, "RELAY_ERROR", str(exc))

    async def _run_resume(
        self, request_id: str, response_id: str, after_cursor: int
    ) -> None:
        if self._response_buffers is None or not response_id:
            logger.info("resume miss (no registry or empty id): %s", response_id)
            await self._send_error(
                request_id, "RESUME_UNKNOWN", "no buffer for this response"
            )
            return
        buf = self._response_buffers.get(response_id)
        if buf is None:
            logger.info("resume miss: response_id=%s", response_id)
            await self._send_error(
                request_id, "RESUME_UNKNOWN", f"no buffer for response_id={response_id}"
            )
            return
        logger.info(
            "resume hit: response_id=%s after_cursor=%d complete=%s buffered=%d",
            response_id, after_cursor, buf.complete, buf.cursor,
        )
        await self._send({
            "type": FRAME_RESPONSE_HEADERS,
            "request_id": request_id,
            "status": buf.status,
            "headers": dict(buf.headers),
        })
        cursor = max(0, after_cursor)
        try:
            while True:
                pending = buf.chunks[cursor:]
                for chunk in pending:
                    await self._send({
                        "type": FRAME_RESPONSE_CHUNK,
                        "request_id": request_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                # Advance only by what this iteration actually emitted.
                # Snapshot-then-send means writers can append more chunks
                # while we're awaiting self._send, so we must NOT jump to
                # buf.cursor — that would skip those concurrent chunks
                # under slow-client / high-throughput conditions. Next
                # iteration picks them up via the buf.chunks[cursor:]
                # snapshot.
                cursor += len(pending)
                if cursor < buf.cursor:
                    continue
                if buf.complete:
                    await self._send({
                        "type": FRAME_RESPONSE_END,
                        "request_id": request_id,
                    })
                    return
                # Wait for the writer to append more (or to finalize on
                # error). The writer always set()s `progressed` after a
                # change, so clearing here is safe — a missed set between
                # drain and clear becomes a no-op next iteration when we
                # re-check `chunks` before waiting again.
                buf.progressed.clear()
                if cursor < buf.cursor or buf.complete:
                    continue
                await buf.progressed.wait()
        except asyncio.CancelledError:
            raise

    async def _serve_blob(self, request_id: str, method: str, path: str) -> None:
        """Stream a registered blob back to the cloud using the standard
        response_headers/response_chunk/response_end framing.

        404 if the token is unknown / expired; 405 if the method isn't GET
        or HEAD (HEAD returns headers only, useful for the client to
        prefetch size/mime without downloading bytes)."""
        if method.upper() not in ("GET", "HEAD"):
            await self._send({
                "type": FRAME_RESPONSE_HEADERS,
                "request_id": request_id,
                "status": 405,
                "headers": {"Allow": "GET, HEAD"},
            })
            await self._send({"type": FRAME_RESPONSE_END, "request_id": request_id})
            return
        token = path[len(BLOB_PATH_PREFIX):]
        assert self._blobs is not None  # narrow Optional for type checkers
        entry = await self._blobs.lookup(token)
        if entry is None:
            await self._send({
                "type": FRAME_RESPONSE_HEADERS,
                "request_id": request_id,
                "status": 404,
                "headers": {"Content-Type": "text/plain; charset=utf-8"},
            })
            await self._send({"type": FRAME_RESPONSE_END, "request_id": request_id})
            return
        try:
            size = entry.path.stat().st_size
        except OSError as exc:
            await self._send_error(request_id, "BLOB_GONE", str(exc))
            return
        response_headers = {
            "Content-Type": entry.mime,
            "Content-Length": str(size),
            "Content-Disposition": f'inline; filename="{entry.filename}"',
            # Browser/<img> caching is fine — blob URLs are content-keyed by
            # token, which is per-image. Capped at the registry TTL so a
            # client doesn't hold a stale entry beyond eviction.
            "Cache-Control": "private, max-age=1800",
        }
        await self._send({
            "type": FRAME_RESPONSE_HEADERS,
            "request_id": request_id,
            "status": 200,
            "headers": response_headers,
        })
        if method.upper() == "HEAD":
            await self._send({"type": FRAME_RESPONSE_END, "request_id": request_id})
            return
        try:
            with entry.path.open("rb") as f:
                while True:
                    chunk = f.read(_BLOB_CHUNK_BYTES)
                    if not chunk:
                        break
                    await self._send({
                        "type": FRAME_RESPONSE_CHUNK,
                        "request_id": request_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
        except OSError as exc:
            await self._send_error(request_id, "BLOB_READ_FAILED", str(exc))
            return
        await self._send({"type": FRAME_RESPONSE_END, "request_id": request_id})

    async def _send_error(self, request_id: str, code: str, message: str) -> None:
        await self._send({
            "type": FRAME_RESPONSE_ERROR,
            "request_id": request_id,
            "code": code,
            "message": message,
        })


class _FrameStreamWriter:
    """Converts in-process dispatcher writes into gateway response frames.

    It also preserves the resumable-response buffering behavior around raw
    response chunks.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        request_id: str,
        response_buffers: Optional[ResponseBufferRegistry],
    ) -> None:
        self._send = send
        self._request_id = request_id
        self._response_buffers = response_buffers
        self._extractor = ResponseIdExtractor() if response_buffers is not None else None
        self._preroll: list[bytes] = []
        self._buffer: Optional[ResponseBuffer] = None
        self._headers: dict[str, str] = {}
        self._status = 0
        self._sent_headers = False
        self._finished = False
        self._progress = _ProgressTracker()

    @property
    def progress(self) -> "_ProgressTracker":
        return self._progress

    async def send_headers(self, status: int, headers: dict[str, str]) -> None:
        if self._sent_headers:
            return
        self._sent_headers = True
        self._status = status
        self._headers = _filter_response_headers(headers)
        await self._send({
            "type": FRAME_RESPONSE_HEADERS,
            "request_id": self._request_id,
            "status": self._status,
            "headers": self._headers,
        })
        self._progress.mark()

    async def send_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        if not self._sent_headers:
            await self.send_headers(200, {})
        self._append_buffer(chunk)
        await self._send({
            "type": FRAME_RESPONSE_CHUNK,
            "request_id": self._request_id,
            "data": base64.b64encode(chunk).decode("ascii"),
        })
        self._progress.mark()

    async def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._sent_headers:
            await self.send_headers(204, {})
        if self._buffer is not None:
            self._buffer.finalize()
        await self._send({"type": FRAME_RESPONSE_END, "request_id": self._request_id})
        self._progress.mark()

    def abandon(self) -> None:
        _abandon_buffer(self._buffer, self._response_buffers)
        self._buffer = None

    def _append_buffer(self, chunk: bytes) -> None:
        if self._extractor is None:
            return
        if self._buffer is not None:
            self._buffer.append(chunk)
            return
        rid = self._extractor.feed(chunk)
        if rid is None:
            self._preroll.append(chunk)
            return
        assert self._response_buffers is not None
        self._buffer = self._response_buffers.create(rid, self._status, self._headers)
        for prev in self._preroll:
            self._buffer.append(prev)
        self._preroll.clear()
        self._buffer.append(chunk)


class _ProgressTracker:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def mark(self) -> None:
        self._version += 1
        self._event.set()

    async def wait_for_change(self, last_seen: int) -> int:
        while self._version == last_seen:
            self._event.clear()
            if self._version != last_seen:
                break
            await self._event.wait()
        return self._version


async def _await_with_progress_timeout(
    coro: Awaitable[int],
    progress: _ProgressTracker,
    *,
    timeout: float,
) -> int:
    task = asyncio.create_task(coro)
    last_seen = progress.version
    progress_task: asyncio.Task[int] | None = None
    try:
        while True:
            if task.done():
                return await task
            progress_task = asyncio.create_task(progress.wait_for_change(last_seen))
            done, _pending = await asyncio.wait(
                {task, progress_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                progress_task.cancel()
                return await task
            if progress_task in done:
                last_seen = progress_task.result()
                progress_task = None
                continue
            progress_task.cancel()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise asyncio.TimeoutError
    except BaseException:
        if progress_task is not None:
            progress_task.cancel()
        if not task.done():
            task.cancel()
        raise


_DROP_REQUEST_HEADERS = {
    # Hop-by-hop.
    "host", "connection", "content-length", "transfer-encoding",
    "upgrade", "keep-alive", "te", "trailer",
    # Browser-only metadata. Hermes's API server treats Origin (and some
    # security-fetch headers) as CSRF signals — forwarding what the browser
    # sent for `localhost:8421` makes Hermes return 403 on perfectly valid
    # requests. Strip them before relaying.
    "origin", "referer", "cookie",
    "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest",
    "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
}


def _abandon_buffer(
    buffer: Optional[ResponseBuffer],
    registry: Optional[ResponseBufferRegistry],
) -> None:
    """On relay error/cancellation, wake any pending resume readers so they
    don't hang on `progressed.wait()`, then drop the entry from the
    registry. The reader will see `complete=True` with no further chunks
    and emit response_end — mobile shows a truncated reply and the user
    falls back to asking Hermes to resend (which has the local log)."""
    if buffer is None or registry is None:
        return
    buffer.finalize()
    registry.drop(buffer.response_id)


def _filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    """Forward only safe, semantically meaningful request headers to Hermes."""
    out = {}
    for k, v in headers.items():
        if k.lower() in _DROP_REQUEST_HEADERS:
            continue
        out[k] = v
    return out


def _filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strip hop-by-hop headers from Hermes's response before forwarding."""
    out = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in {"connection", "transfer-encoding", "content-length"}:
            continue
        out[k] = v
    return out
