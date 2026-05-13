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
from typing import Any

from .client import HandshakeError, VylenGatewayClient
from .config import ConfigError, load_from_env
from .health import HealthReporter
from .relay import FRAME_REQUEST, HermesRelay

logger = logging.getLogger(__name__)


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

                async def on_frame(frame):
                    if frame.get("type") == FRAME_REQUEST:
                        await relay.handle(frame)

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
            self._relay = HermesRelay(client.send)
            self._health = HealthReporter(
                client.send,
                hermes_url=self._relay.hermes_url,
                hermes_api_key=os.environ.get("VYLEN_HERMES_API_KEY") or None,
            )
            self._health.start()
            logger.info(
                "Vylen gateway online: instance_id=%s user_id=%s hermes=%s",
                ready.instance_id, ready.user_id, self._relay.hermes_url,
            )
            return True

        async def _teardown_session(self) -> None:
            if self._health:
                await self._health.stop()
                self._health = None
            if self._relay:
                await self._relay.close()
                self._relay = None
            if self._client:
                await self._client.close()
                self._client = None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            # Checkpoint 4 fills this in. For now, signal "not yet wired" so
            # any caller during the handshake-only phase gets a clean error
            # rather than silent success.
            from gateway.platforms.base import SendResult
            return SendResult(success=False, error="vylen gateway: send() not implemented until checkpoint 4")

        async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
            return {"name": "vylen", "type": "dm"}

    return VylenGatewayAdapter


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
