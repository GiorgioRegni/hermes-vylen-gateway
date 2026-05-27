"""Async WebSocket client that speaks the Vylen gateway protocol.

Standalone of Hermes — both the adapter (production path) and the doctor CLI
(standalone diagnostic) use this same class.
"""

from __future__ import annotations

import asyncio
import importlib.metadata as importlib_metadata
import json
import logging
import platform as _platform
import socket
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection, connect as ws_connect

from .config import GatewayConfig

logger = logging.getLogger(__name__)

# Frame type constants — kept in sync with the Go cloud (see
# cloud/internal/cloud/gateway.go).
FRAME_HELLO = "hello"
FRAME_READY = "ready"
FRAME_ERROR = "error"

PLUGIN_VERSION = "0.1.0"


def _detect_hermes_version() -> Optional[str]:
    try:
        return importlib_metadata.version("hermes-agent")
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass
class ReadyInfo:
    instance_id: str
    user_id: str
    server_time: str
    relay_id: str = ""
    relay_generation: str = ""
    relay_region: str = ""


@dataclass
class HelloMeta:
    plugin_version: str = PLUGIN_VERSION
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=_platform.python_version)
    hermes_version: Optional[str] = field(default_factory=_detect_hermes_version)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "plugin_version": self.plugin_version,
            "hostname": self.hostname,
            "python_version": self.python_version,
        }
        if self.hermes_version:
            d["hermes_version"] = self.hermes_version
        d.update(self.extra)
        return d


class HandshakeError(Exception):
    """Raised when the cloud does not complete a clean hello/ready exchange."""


class VylenGatewayClient:
    """Owns a single long-lived WebSocket to the cloud.

    `connect()` performs the handshake and returns the `ready` payload. After
    that, callers can pump frames via `iter_frames()` and write via `send()`.
    Reconnection is the caller's responsibility (the adapter wraps this with a
    backoff loop in a later checkpoint).
    """

    def __init__(self, config: GatewayConfig, meta: HelloMeta | None = None):
        self._config = config
        self._meta = meta or HelloMeta()
        self._conn: ClientConnection | None = None

    async def connect(self, *, timeout: float = 10.0) -> ReadyInfo:
        headers = {"Authorization": self._config.authorization_header}
        logger.info("Dialing %s", self._config.websocket_url)
        try:
            conn = await asyncio.wait_for(
                ws_connect(
                    self._config.websocket_url,
                    additional_headers=headers,
                    # Cloud accepts Hermes / transcribe bodies up to 10MB
                    # and forwards them base64-encoded inside a single
                    # `request` / `transcribe` frame. Base64 expansion is
                    # 4/3, plus JSON envelope overhead — round to 16MB so
                    # legitimate large image/audio payloads aren't rejected
                    # at the WS layer with "message too big". Keep in sync
                    # with the cloud's body cap in
                    # cloud/internal/cloud/server.go.
                    max_size=16 * 1024 * 1024,
                    open_timeout=timeout,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise HandshakeError(f"timeout dialing {self._config.websocket_url}") from exc
        except websockets.InvalidStatus as exc:  # type: ignore[attr-defined]
            raise HandshakeError(
                f"cloud rejected the connection: HTTP {exc.response.status_code}"
            ) from exc
        except OSError as exc:
            raise HandshakeError(f"failed dialing {self._config.websocket_url}: {exc}") from exc
        self._conn = conn

        hello = {
            "type": FRAME_HELLO,
            "instance_meta": self._meta.to_dict(),
        }
        await self._conn.send(json.dumps(hello))

        try:
            raw = await asyncio.wait_for(self._conn.recv(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self._conn.close()
            self._conn = None
            raise HandshakeError("no ready frame from cloud within timeout") from exc

        frame = _parse_frame(raw)
        if frame.get("type") == FRAME_ERROR:
            msg = frame.get("message") or frame.get("code") or "cloud rejected hello"
            raise HandshakeError(f"cloud error: {msg}")
        if frame.get("type") != FRAME_READY:
            raise HandshakeError(f"expected type=ready, got {frame!r}")

        instance_id = frame.get("instance_id") or ""
        user_id = frame.get("user_id") or ""
        server_time = frame.get("server_time") or ""
        relay_id = frame.get("relay_id") or ""
        relay_generation = frame.get("relay_generation") or ""
        relay_region = frame.get("relay_region") or ""
        if not instance_id:
            raise HandshakeError("ready frame missing instance_id")
        logger.info(
            "Handshake OK: instance_id=%s user_id=%s relay=%s generation=%s",
            instance_id,
            user_id,
            relay_id,
            relay_generation,
        )
        return ReadyInfo(
            instance_id=instance_id,
            user_id=user_id,
            server_time=server_time,
            relay_id=relay_id,
            relay_generation=relay_generation,
            relay_region=relay_region,
        )

    async def iter_frames(
        self,
        on_frame: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Pump frames until the socket closes. No-op handler if `on_frame` is None.

        Checkpoint 4 will fill this in with real frame dispatch.
        """
        if self._conn is None:
            raise RuntimeError("connect() must be called first")
        async for raw in self._conn:
            frame = _parse_frame(raw)
            if on_frame:
                await on_frame(frame)

    async def send(self, frame: dict[str, Any]) -> None:
        if self._conn is None:
            raise RuntimeError("connect() must be called first")
        await self._conn.send(json.dumps(frame))

    async def close(self) -> None:
        if self._conn is None:
            return
        try:
            await self._conn.close()
        finally:
            self._conn = None


def _parse_frame(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
