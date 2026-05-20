"""Chat-level cursor subscribe/replay support for retained chat events."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
import uuid
from typing import Any, Awaitable, Callable

from .event_log import EventLogRegistry, ResumeExpired

FRAME_CHAT_SUBSCRIBE = "chat_subscribe"
FRAME_CHAT_UNSUBSCRIBE = "chat_unsubscribe"
FRAME_CHAT_EVENT = "chat_event"
FRAME_CHAT_RESUME_EXPIRED = "chat_resume_expired"

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class ChatCursorRelay:
    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        logs: EventLogRegistry,
        *,
        disabled: bool = False,
    ) -> None:
        self._send = send
        self._logs = logs
        self._disabled = disabled
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._push_lock = asyncio.Lock()

    def append_push(self, frame: dict[str, Any]) -> int | None:
        if self._disabled:
            return None
        chat_id = _clean_id(str(frame.get("chat_id") or ""))
        if not chat_id:
            return None
        frame.setdefault("event_id", _event_id())
        return self.append_event(chat_id, "push", dict(frame))

    def append_event(self, chat_id: str, kind: str, payload: dict[str, Any]) -> int | None:
        if self._disabled:
            return None
        clean_chat_id = _clean_id(str(chat_id or ""))
        if not clean_chat_id or not kind:
            return None
        event = self._logs.get_or_create(clean_chat_id).append(kind, dict(payload))
        return event.seq

    async def send_event(self, chat_id: str, kind: str, payload: dict[str, Any]) -> int | None:
        return self.append_event(chat_id, kind, payload)

    async def send_push(self, frame: dict[str, Any]) -> int | None:
        if self._disabled:
            await self._send(frame)
            return None
        chat_id = _clean_id(str(frame.get("chat_id") or ""))
        if not chat_id:
            await self._send(frame)
            return None
        async with self._push_lock:
            existing_log = self._logs.get(chat_id)
            log = existing_log or self._logs.get_or_create(chat_id)
            seq = log.next_seq
            frame["seq"] = seq
            frame.setdefault("event_id", _event_id())
            log.ensure_fits("push", dict(frame))
            try:
                await self._send(frame)
            except Exception:
                if existing_log is None and not log.events:
                    self._logs.drop(chat_id)
                raise
            event = log.append("push", dict(frame))
            return event.seq

    async def handle_subscribe(self, frame: dict[str, Any]) -> None:
        request_id = _clean_id(str(frame.get("request_id") or ""))
        chat_id = _clean_id(str(frame.get("chat_id") or ""))
        client_id = _clean_id(str(frame.get("client_id") or ""))
        after_seq = _parse_seq(frame.get("after_seq"))
        if not request_id:
            return
        if self._disabled:
            await self._send({
                "type": FRAME_CHAT_RESUME_EXPIRED,
                "request_id": request_id,
                "chat_id": chat_id,
                "code": "CHAT_CURSORS_DISABLED",
                "floor_seq": 0,
                "latest_seq": 0,
            })
            return
        if not chat_id or not client_id:
            await self._send({
                "type": FRAME_CHAT_RESUME_EXPIRED,
                "request_id": request_id,
                "chat_id": chat_id,
                "code": "CHAT_SUBSCRIBE_INVALID",
                "floor_seq": 0,
                "latest_seq": 0,
            })
            return
        self.cancel(request_id)
        self._tasks[request_id] = asyncio.create_task(
            self._tail(request_id, chat_id, client_id, after_seq)
        )

    def cancel(self, request_id: str) -> None:
        task = self._tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _tail(self, request_id: str, chat_id: str, client_id: str, after_seq: int) -> None:
        log = self._logs.get_or_create(chat_id)
        try:
            async for event in log.tail_after(
                after_seq,
                keepalive_seconds=None,
                reject_future_cursor=True,
            ):
                await self._send({
                    "type": FRAME_CHAT_EVENT,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "seq": event.seq,
                    "occurred_at": _iso_time(event.created_at),
                    "event": {
                        "kind": event.kind,
                        "payload": event.payload,
                    },
                })
                self._logs.acknowledge(chat_id, client_id, event.seq)
        except ResumeExpired as exc:
            await self._send({
                "type": FRAME_CHAT_RESUME_EXPIRED,
                "request_id": request_id,
                "chat_id": chat_id,
                "code": "CHAT_RESUME_EXPIRED",
                "floor_seq": exc.floor_seq,
                "latest_seq": exc.latest_seq,
            })
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if self._tasks.get(request_id) is current:
                self._tasks.pop(request_id, None)


def _clean_id(value: str) -> str:
    value = value.strip()
    if not _ID_RE.match(value):
        return ""
    return value


def _parse_seq(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _iso_time(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat().replace("+00:00", "Z")
