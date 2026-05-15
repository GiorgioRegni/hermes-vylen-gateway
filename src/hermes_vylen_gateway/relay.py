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
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Frame type constants — keep in sync with cloud/internal/cloud/gateway.go.
FRAME_REQUEST = "request"
FRAME_RESPONSE_HEADERS = "response_headers"
FRAME_RESPONSE_CHUNK = "response_chunk"
FRAME_RESPONSE_END = "response_end"
FRAME_RESPONSE_ERROR = "response_error"

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
    ):
        self._send = send_frame
        self._hermes_url = (hermes_url or os.environ.get("VYLEN_HERMES_URL", DEFAULT_HERMES_URL)).rstrip("/")
        self._hermes_api_key = (hermes_api_key or os.environ.get("VYLEN_HERMES_API_KEY", "")).strip()
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
        try:
            async with self._client.stream(
                method, url, content=body, headers=forwarded
            ) as resp:
                await self._send({
                    "type": FRAME_RESPONSE_HEADERS,
                    "request_id": request_id,
                    "status": resp.status_code,
                    "headers": _filter_response_headers(dict(resp.headers)),
                })
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    await self._send({
                        "type": FRAME_RESPONSE_CHUNK,
                        "request_id": request_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                await self._send({
                    "type": FRAME_RESPONSE_END,
                    "request_id": request_id,
                })
        except httpx.HTTPError as exc:
            logger.warning("relay error for %s %s: %s", method, url, exc)
            await self._send_error(request_id, "HERMES_UNREACHABLE", str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("relay crashed for %s %s", method, url)
            await self._send_error(request_id, "RELAY_ERROR", str(exc))

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
