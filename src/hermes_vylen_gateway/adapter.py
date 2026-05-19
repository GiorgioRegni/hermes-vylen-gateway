"""VylenGatewayAdapter — a Hermes BasePlatformAdapter that proxies between
the agent and Vylen Cloud over the gateway WebSocket.

This module imports from `hermes_agent.*` lazily inside the class body so the
package itself remains importable when Hermes is not installed (the doctor CLI
relies on this).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional

from .blobs import BlobRegistry
from .chat_cursor import (
    FRAME_CHAT_SUBSCRIBE,
    FRAME_CHAT_UNSUBSCRIBE,
    ChatCursorRelay,
)
from .client import HandshakeError, VylenGatewayClient
from .config import ConfigError, load_from_env
from .event_log import EventLogRegistry
from .health import HealthReporter
from .memory import FRAME_MEMORY_REQUEST, MemoryRPC
from .relay import FRAME_REQUEST, FRAME_RESPONSE_RESUME, HermesRelay
from .response_buffer import ResponseBufferRegistry
from .transcribe import FRAME_TRANSCRIBE, Transcriber

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s", name, raw, default)
        return default


async def _sweep_loop(
    registry: Any, interval_seconds: float, *, label: str = "registry"
) -> None:
    """Periodically drop completed/expired in-memory gateway state."""
    if interval_seconds <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise
        try:
            evicted = registry.sweep()
            if evicted:
                logger.debug("%s sweep: evicted=%d", label, evicted)
        except Exception:  # noqa: BLE001
            logger.exception("%s sweep failed", label)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %s", name, raw, default)
        return default

# Cron output reaches `BasePlatformAdapter.send()` wrapped in this envelope
# (built by `cron/scheduler.py` around line 503; controlled by the
# `cron.wrap_response` config key, on by default). We strip it before
# forwarding to the Vylen UI and extract the structured `job_id`/`name`
# so the reply path can label and attribute pushes.
#
# Shape (job_id + name are required; the "To stop" footer is best-effort):
#   Cronjob Response: <name>
#   (job_id: <id>)
#   -------------
#
#   <actual content>
#
#   To stop or manage this job, send me a new message ...
#
# Tracked as backlog: ask upstream Hermes to pass these as `metadata`
# instead of regex-parsing. See vylen/docs/invariants.md.
_CRON_ENVELOPE_RE = re.compile(
    r"\ACronjob Response:\s*(?P<name>.*?)\n"
    r"\(job_id:\s*(?P<job_id>[^\)]+)\)\n"
    r"-+\n+"
    r"(?P<body>.*?)"
    r"(?:\n+To stop or manage this job[^\n]*)?\Z",
    re.DOTALL,
)


def _import_hermes():
    """Lazy import the Hermes pieces we extend. Raises ImportError if missing."""
    from gateway.platforms.base import BasePlatformAdapter, Platform  # noqa: F401
    return BasePlatformAdapter, Platform


def make_adapter_class():
    """Build the adapter class lazily so module import doesn't require Hermes."""
    BasePlatformAdapter, Platform = _import_hermes()

    class VylenGatewayAdapter(BasePlatformAdapter):
        """Hermes side of the Vylen Cloud gateway WebSocket.

        Checkpoint 3 implements the connect / disconnect / handshake path
        only. send() is a stub that will be filled in at checkpoint 4 when
        message routing lands.
        """

        def __init__(self, config, platform=None):
            # Platform("vylen") goes through Platform._missing_ which returns
            # the pseudo-member the platform_registry already created when
            # register(ctx) ran. Identity-stable across calls.
            super().__init__(config, platform or Platform("vylen"))
            self._client: VylenGatewayClient | None = None
            self._task: asyncio.Task | None = None
            self._instance_id: str | None = None
            self._relay: HermesRelay | None = None
            self._health: HealthReporter | None = None
            self._transcribe: Transcriber | None = None
            self._memory: MemoryRPC | None = None
            self._chat_cursors: ChatCursorRelay | None = None
            self._chat_event_logs = EventLogRegistry(
                ttl_seconds=_env_float("VYLEN_CHAT_CURSOR_TTL_SECONDS", 900.0),
                max_events=_env_int("VYLEN_CHAT_CURSOR_MAX_EVENTS", 1000),
                max_bytes=_env_int("VYLEN_CHAT_CURSOR_MAX_BYTES", 4 * 1024 * 1024),
            )
            # One blob registry per adapter lifetime; entries auto-expire
            # (see blobs.py). Resets implicitly across reconnects since the
            # adapter is reconstructed and the cloud only references tokens
            # from the current session anyway.
            self._blobs: BlobRegistry | None = None
            self._response_buffers: ResponseBufferRegistry | None = None
            self._response_sweep_task: asyncio.Task | None = None
            self._chat_sweep_task: asyncio.Task | None = None
            self._stopping = False

        async def connect(self) -> bool:
            try:
                load_from_env()
            except ConfigError as exc:
                logger.error("Vylen gateway config invalid: %s", exc)
                return False
            self._stopping = False
            # Start the supervisor; it owns the WS lifecycle and reconnects
            # the socket on every drop. Initial dial happens in the loop so
            # connect() returns True immediately even if the cloud is briefly
            # unreachable at boot.
            self._task = asyncio.create_task(self._supervisor())
            return True

        async def disconnect(self) -> None:
            self._stopping = True
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._task = None
            await self._teardown_session()

        async def _supervisor(self) -> None:
            backoff = 1.0
            while not self._stopping:
                if not await self._open_session():
                    # Failed to dial. Backoff up to 60s.
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 60.0)
                    continue
                backoff = 1.0
                # Pump frames until the socket dies, then loop and reconnect.
                assert self._client is not None and self._relay is not None
                relay = self._relay
                transcriber = self._transcribe
                memory = self._memory

                async def on_frame(frame):
                    t = frame.get("type")
                    if t == FRAME_REQUEST:
                        await relay.handle(frame)
                    elif t == FRAME_RESPONSE_RESUME:
                        await relay.handle_resume(frame)
                    elif t == FRAME_CHAT_SUBSCRIBE and self._chat_cursors is not None:
                        await self._chat_cursors.handle_subscribe(frame)
                    elif t == FRAME_CHAT_UNSUBSCRIBE and self._chat_cursors is not None:
                        self._chat_cursors.cancel(str(frame.get("request_id") or ""))
                    elif t == FRAME_TRANSCRIBE and transcriber is not None:
                        await transcriber.handle(frame)
                    elif t == FRAME_MEMORY_REQUEST and memory is not None:
                        await memory.handle(frame)

                try:
                    await self._client.iter_frames(on_frame)
                except Exception as exc:  # noqa: BLE001
                    logger.info("Vylen gateway socket dropped: %s", exc)
                await self._teardown_session()
                if self._stopping:
                    return
                logger.info("Vylen gateway reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)

        async def _open_session(self) -> bool:
            try:
                gateway_cfg = load_from_env()
            except ConfigError as exc:
                logger.error("Vylen gateway config invalid: %s", exc)
                return False
            client = VylenGatewayClient(gateway_cfg)
            try:
                ready = await client.connect()
            except HandshakeError as exc:
                logger.warning("Vylen gateway handshake failed: %s", exc)
                await client.close()
                return False
            self._client = client
            self._instance_id = ready.instance_id
            self._blobs = BlobRegistry()
            self._response_buffers = ResponseBufferRegistry(
                grace_seconds=_env_float("VYLEN_RESUME_GRACE_SECONDS", 300.0),
                max_bytes=_env_int("VYLEN_RESUME_MAX_BYTES", 4 * 1024 * 1024),
            )
            sweep_interval = _env_float("VYLEN_RESUME_SWEEP_SECONDS", 60.0)
            self._response_sweep_task = asyncio.create_task(
                _sweep_loop(self._response_buffers, sweep_interval, label="response buffer")
            )
            chat_sweep_interval = _env_float("VYLEN_CHAT_CURSOR_SWEEP_SECONDS", 60.0)
            self._chat_sweep_task = asyncio.create_task(
                _sweep_loop(self._chat_event_logs, chat_sweep_interval, label="chat event log")
            )
            self._relay = HermesRelay(
                client.send,
                blobs=self._blobs,
                response_buffers=self._response_buffers,
            )
            self._chat_cursors = ChatCursorRelay(
                client.send,
                self._chat_event_logs,
                disabled=bool(os.environ.get("VYLEN_CHAT_CURSOR_DISABLE")),
            )
            self._health = HealthReporter(client.send)
            self._health.start()
            self._transcribe = Transcriber(client.send)
            self._memory = MemoryRPC(client.send)
            logger.info(
                "Vylen gateway online: instance_id=%s user_id=%s hermes=in-process",
                ready.instance_id, ready.user_id,
            )
            return True

        async def _teardown_session(self) -> None:
            if self._response_sweep_task:
                self._response_sweep_task.cancel()
                try:
                    await self._response_sweep_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._response_sweep_task = None
            if self._chat_sweep_task:
                self._chat_sweep_task.cancel()
                try:
                    await self._chat_sweep_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._chat_sweep_task = None
            if self._memory:
                await self._memory.close()
                self._memory = None
            if self._transcribe:
                await self._transcribe.close()
                self._transcribe = None
            if self._health:
                await self._health.stop()
                self._health = None
            if self._relay:
                await self._relay.close()
                self._relay = None
            if self._chat_cursors:
                await self._chat_cursors.close()
                self._chat_cursors = None
            # BlobRegistry holds no OS resources (just an in-memory map);
            # drop the reference so any outstanding tokens immediately
            # become unaddressable across reconnects.
            self._blobs = None
            self._response_buffers = None
            if self._client:
                await self._client.close()
                self._client = None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            # Plugin-initiated message — typically a Hermes cron with
            # `deliver=vylen`, or `BasePlatformAdapter.send_image` etc.
            # falling back to `.send`. We emit a `push` frame; the cloud
            # fans out to any SSE subscribers on /v1/notifications for the
            # owning user. No retries, no persistence — foreground pilot
            # only. Background delivery (FCM) ships behind the Firebase
            # Auth follow-up.
            from gateway.platforms.base import SendResult
            if self._client is None:
                return SendResult(
                    success=False,
                    error="vylen gateway: socket not connected",
                    retryable=True,
                )
            raw = content if isinstance(content, str) else str(content)
            body, cron_job_id, cron_job_name = _parse_cron_envelope(raw)
            frame: dict[str, Any] = {
                "type": "push",
                "chat_id": str(chat_id) if chat_id is not None else "",
                "text": body,
            }
            if cron_job_id:
                frame["cron_job_id"] = cron_job_id
            if cron_job_name:
                frame["cron_job_name"] = cron_job_name
            try:
                if self._chat_cursors is not None:
                    await self._chat_cursors.send_push(frame)
                else:
                    await self._client.send(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vylen gateway: push frame send failed: %s", exc)
                return SendResult(success=False, error=str(exc), retryable=True)
            return SendResult(success=True)

        async def send_image_file(
            self,
            chat_id: str,
            image_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
            **kwargs,
        ):
            # Tunnel-streamed media: register the local file in the
            # BlobRegistry, get a short-lived token, ship just that on the
            # push frame. The client fetches the bytes via
            # /v1/instances/<id>/blobs/<token> on the cloud, which tunnels
            # the read through the existing gateway WS (see relay._serve_blob).
            # Push frames stay small; the SSE channel doesn't carry image
            # bytes; multiple clients reuse the same URL.
            from gateway.platforms.base import SendResult
            if self._client is None or self._blobs is None:
                return SendResult(
                    success=False,
                    error="vylen gateway: socket not connected",
                    retryable=True,
                )
            registered = await self._blobs.register(image_path)
            if registered is None:
                logger.warning("vylen gateway: could not register image %s", image_path)
                # Fall back to the base class behaviour (path-as-text) so the
                # user at least sees that an image was attempted, rather than
                # silent dropping.
                return await super(VylenGatewayAdapter, self).send_image_file(
                    chat_id=chat_id,
                    image_path=image_path,
                    caption=caption,
                    reply_to=reply_to,
                    metadata=metadata,
                    **kwargs,
                )
            token, mime, filename = registered
            raw_caption = caption or ""
            body, cron_job_id, cron_job_name = _parse_cron_envelope(raw_caption)
            frame: dict[str, Any] = {
                "type": "push",
                "chat_id": str(chat_id) if chat_id is not None else "",
                "text": body,
                "image_token": token,
                "image_mime": mime,
                "image_filename": filename,
            }
            if cron_job_id:
                frame["cron_job_id"] = cron_job_id
            if cron_job_name:
                frame["cron_job_name"] = cron_job_name
            try:
                if self._chat_cursors is not None:
                    await self._chat_cursors.send_push(frame)
                else:
                    await self._client.send(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vylen gateway: image push frame send failed: %s", exc)
                return SendResult(success=False, error=str(exc), retryable=True)
            return SendResult(success=True)

        async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
            return {"name": "vylen", "type": "dm"}

    return VylenGatewayAdapter


def _parse_cron_envelope(text: str) -> tuple[str, str, str]:
    """Strip Hermes's cron-response chrome from `text` and extract job_id +
    job_name. Returns `(body, job_id, job_name)`. If the envelope doesn't
    match (chat message, ad-hoc send, or upstream changed the format),
    returns the raw text and empty strings for the IDs — preserving the
    pre-cron behavior so chat sends still work.
    """
    match = _CRON_ENVELOPE_RE.match(text)
    if not match:
        return text, "", ""
    body = match.group("body").strip()
    return body, match.group("job_id").strip(), match.group("name").strip()


def adapter_factory(config):
    """Hermes calls this with a PlatformConfig and expects an adapter instance."""
    cls = make_adapter_class()
    return cls(config)


def check_dependencies() -> bool:
    """Hermes calls this before instantiation to verify deps. We need:
    - VYLEN_INSTANCE_TOKEN set
    - websockets importable (it's our own dep so it always is when we are)
    """
    try:
        load_from_env()
    except ConfigError as exc:
        logger.info("Vylen gateway not configured: %s", exc)
        return False
    return True
