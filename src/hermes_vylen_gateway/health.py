"""Periodic Hermes health reporter.

Without this, the cloud has no way to learn that local in-process Hermes
dispatch is usable and so the instance pill in mobile/web shows "Degraded"
forever (instance is WS-connected but hermes_status remains "unknown").
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

from .agent_runner import check_available

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_INTERVAL_S = 15.0

FRAME_HEARTBEAT = "heartbeat"
FRAME_HEALTH = "health"


class HealthReporter:
    """Background task that periodically probes local Hermes and emits a
    `health` frame so the cloud's `hermes_status` stays accurate."""

    def __init__(
        self,
        send_frame: Callable[[dict[str, Any]], Awaitable[None]],
        interval_s: float | None = None,
        chat_state_status: Callable[[], Any] | None = None,
    ):
        self._send = send_frame
        self._interval = interval_s if interval_s is not None else _interval_from_env()
        self._chat_state_status = chat_state_status
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="vylen-health-reporter")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._task = None

    async def _run(self) -> None:
        # Send the first probe immediately so the cloud flips off "Degraded"
        # within seconds of connect rather than waiting a full interval.
        await self._probe_and_send()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return  # stop signaled
            except asyncio.TimeoutError:
                pass
            await self._probe_and_send()

    async def _probe_and_send(self) -> None:
        status, latency_ms, last_error = await self._probe()
        frame: dict[str, Any] = {
            "type": FRAME_HEALTH,
            "hermes_status": status,
            "hermes_latency_ms": latency_ms,
        }
        if last_error is not None:
            frame["last_error"] = last_error
        if self._chat_state_status is not None:
            try:
                chat_status = self._chat_state_status()
                status_text = str(getattr(chat_status, "status", "") or "")
                message = getattr(chat_status, "message", None)
                if status_text:
                    frame["chat_state_status"] = status_text
                if message:
                    frame["chat_state_message"] = _sanitize_chat_state_message(status_text)
                if status_text and status_text not in {"ok", "initializing"} and last_error is None:
                    frame["last_error"] = f"chat_state: {status_text}"
            except Exception:  # noqa: BLE001
                logger.debug("chat state status probe failed", exc_info=True)
        try:
            await self._send(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("health send failed: %s", exc)

    async def _probe(self) -> tuple[str, int, str | None]:
        started = time.perf_counter()
        ok, error = check_available()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if ok:
            return "ok", elapsed_ms, None
        return "unreachable", elapsed_ms, error


def _sanitize_chat_state_message(status: str) -> str:
    if status == "reset_after_corruption":
        return "Local Vylen chat state was reset after SQLite corruption."
    if status == "version_mismatch":
        return "Local Vylen chat state schema is newer than this plugin supports."
    if status == "open_failed":
        return "Local Vylen chat state is unavailable."
    return str(status or "chat_state_unavailable")


def _interval_from_env() -> float:
    raw = os.environ.get("VYLEN_HEALTH_INTERVAL_S", "")
    try:
        v = float(raw) if raw else DEFAULT_HEALTH_INTERVAL_S
    except ValueError:
        return DEFAULT_HEALTH_INTERVAL_S
    return max(1.0, v)
