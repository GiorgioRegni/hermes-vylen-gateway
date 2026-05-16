"""HTTP-tunnel relay: bridges gateway WS request frames to a local Hermes
HTTP API, streaming the response chunks back as WS response frames.

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

import httpx

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

DEFAULT_HERMES_URL = "http://127.0.0.1:8000"


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
        hermes_url: str | None = None,
        hermes_api_key: str | None = None,
        request_timeout: float | None = None,
        blobs: Optional[BlobRegistry] = None,
        response_buffers: Optional[ResponseBufferRegistry] = None,
    ):
        self._send = send_frame
        self._hermes_url = (hermes_url or os.environ.get("VYLEN_HERMES_URL", DEFAULT_HERMES_URL)).rstrip("/")
        self._hermes_api_key = (hermes_api_key or os.environ.get("VYLEN_HERMES_API_KEY", "")).strip()
        # Shared blob registry — populated by the adapter when Hermes
        # produces an image; served by `_serve_blob` when the cloud tunnels
        # a request to `/__vylen_blob__/<token>`.
        self._blobs = blobs
        # Response buffer registry. Optional so callers that don't need
        # resume (tests, blob-only mocks) can omit it; the adapter wires a
        # real registry in production. See spec 003.
        self._response_buffers = response_buffers
        # Bound the local Hermes call lifetime so a hung backend (socket
        # accepted but no response bytes, model server stalled, etc.)
        # can't accumulate background tasks indefinitely — cloud-side
        # timeouts only stop *waiting* for the response, they don't
        # cancel plugin work. The default (5 minutes total per request,
        # 30s to receive the first byte) is generous enough for the
        # longest tool-using LLM runs but finite. Override per-test via
        # the constructor arg or globally via `VYLEN_HERMES_TIMEOUT`
        # (seconds; "none" disables the bound — only for diagnostics).
        if request_timeout is None:
            request_timeout = _resolve_request_timeout()
        if request_timeout is None:
            timeout: httpx.Timeout | None = None
        else:
            # Generous read window so streaming /v1/responses runs can
            # span minutes between SSE chunks; tight connect/pool/write so
            # network-level stalls fail fast.
            timeout = httpx.Timeout(request_timeout, connect=30.0, pool=10.0, write=30.0)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._tasks: set[asyncio.Task] = set()

    @property
    def hermes_url(self) -> str:
        return self._hermes_url

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
        await self._client.aclose()

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

        url = self._hermes_url + path
        forwarded = _filter_request_headers(headers)
        if self._hermes_api_key and "authorization" not in {k.lower() for k in forwarded}:
            # Local Hermes API is bearer-authenticated in compose setups.
            # Surface the key from env so the plugin can satisfy that
            # without the upstream client knowing about it.
            forwarded["Authorization"] = f"Bearer {self._hermes_api_key}"
        logger.debug("relay -> %s %s (request_id=%s)", method, url, request_id)
        # Resume buffering state. We don't know the Hermes response_id until
        # the `response.created` SSE event arrives, so we pre-buffer chunks
        # in `preroll` and graduate to a real buffer once the id is found.
        # If the id never arrives (non-SSE response, error, etc.) we just
        # drop the preroll and the response is non-resumable.
        extractor: Optional[ResponseIdExtractor] = (
            ResponseIdExtractor() if self._response_buffers is not None else None
        )
        preroll: list[bytes] = []
        buffer: Optional[ResponseBuffer] = None
        seen_headers: dict[str, str] = {}
        seen_status: int = 0
        try:
            async with self._client.stream(
                method, url, content=body, headers=forwarded
            ) as resp:
                seen_status = resp.status_code
                seen_headers = _filter_response_headers(dict(resp.headers))
                await self._send({
                    "type": FRAME_RESPONSE_HEADERS,
                    "request_id": request_id,
                    "status": seen_status,
                    "headers": seen_headers,
                })
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    await self._send({
                        "type": FRAME_RESPONSE_CHUNK,
                        "request_id": request_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                    if extractor is None:
                        continue
                    if buffer is not None:
                        buffer.append(chunk)
                        continue
                    rid = extractor.feed(chunk)
                    if rid is None:
                        preroll.append(chunk)
                        continue
                    assert self._response_buffers is not None
                    buffer = self._response_buffers.create(rid, seen_status, seen_headers)
                    for prev in preroll:
                        buffer.append(prev)
                    preroll.clear()
                    buffer.append(chunk)
                if buffer is not None:
                    buffer.finalize()
                await self._send({
                    "type": FRAME_RESPONSE_END,
                    "request_id": request_id,
                })
        except httpx.HTTPError as exc:
            _abandon_buffer(buffer, self._response_buffers)
            logger.warning("relay error for %s %s: %s", method, url, exc)
            await self._send_error(request_id, "HERMES_UNREACHABLE", str(exc))
        except asyncio.CancelledError:
            _abandon_buffer(buffer, self._response_buffers)
            raise
        except Exception as exc:  # noqa: BLE001
            _abandon_buffer(buffer, self._response_buffers)
            logger.exception("relay crashed for %s %s", method, url)
            await self._send_error(request_id, "RELAY_ERROR", str(exc))

    async def _run_resume(
        self, request_id: str, response_id: str, after_cursor: int
    ) -> None:
        if self._response_buffers is None or not response_id:
            await self._send_error(
                request_id, "RESUME_UNKNOWN", "no buffer for this response"
            )
            return
        buf = self._response_buffers.get(response_id)
        if buf is None:
            await self._send_error(
                request_id, "RESUME_UNKNOWN", f"no buffer for response_id={response_id}"
            )
            return
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
                cursor = buf.cursor
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
                if buf.cursor > cursor or buf.complete:
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
