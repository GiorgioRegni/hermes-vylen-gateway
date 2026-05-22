"""Chat-level cursor subscribe/replay support for retained chat events."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
import uuid
from typing import Any, Awaitable, Callable

from .chat_store import ChatStateUnavailable, InvalidChatStateEvent
from .event_log import EventLogRegistry, ResumeExpired

FRAME_CHAT_SUBSCRIBE = "chat_subscribe"
FRAME_CHAT_UNSUBSCRIBE = "chat_unsubscribe"
FRAME_CHAT_EVENT = "chat_event"
FRAME_CHAT_RESUME_EXPIRED = "chat_resume_expired"
FRAME_CHAT_LIST = "chat_list"
FRAME_CHAT_LIST_RESPONSE = "chat_list_response"
FRAME_CHAT_LIST_ERROR = "chat_list_error"
FRAME_CHAT_SNAPSHOT = "chat_snapshot"
FRAME_CHAT_SNAPSHOT_RESPONSE = "chat_snapshot_response"
FRAME_CHAT_SNAPSHOT_ERROR = "chat_snapshot_error"

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
        return await asyncio.to_thread(self.append_event, chat_id, kind, payload)

    async def send_push(self, frame: dict[str, Any]) -> int | None:
        if self._disabled:
            await self._send(frame)
            return None
        chat_id = _clean_id(str(frame.get("chat_id") or ""))
        if not chat_id:
            await self._send(frame)
            return None
        async with self._push_lock:
            existing_log = await asyncio.to_thread(self._logs.get, chat_id)
            log = existing_log or await asyncio.to_thread(self._logs.get_or_create, chat_id)
            seq = await asyncio.to_thread(lambda: log.next_seq)
            frame["seq"] = seq
            frame.setdefault("event_id", _event_id())
            await asyncio.to_thread(log.ensure_fits, "push", dict(frame))
            try:
                await self._send(frame)
            except Exception:
                if existing_log is None and not await asyncio.to_thread(lambda: bool(log.events)):
                    await asyncio.to_thread(self._logs.drop, chat_id)
                raise
            event = await asyncio.to_thread(log.append, "push", dict(frame))
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

    async def handle_list(self, frame: dict[str, Any]) -> None:
        request_id = _clean_id(str(frame.get("request_id") or ""))
        if not request_id:
            return
        if self._disabled:
            await self._send_error(FRAME_CHAT_LIST_ERROR, request_id, "CHAT_CURSORS_DISABLED")
            return
        if not hasattr(self._logs, "list_chats"):
            await self._send_error(FRAME_CHAT_LIST_ERROR, request_id, "CHAT_STATE_UNAVAILABLE")
            return
        try:
            page = await asyncio.to_thread(
                self._logs.list_chats,
                limit=_parse_limit(frame.get("limit"), default=50, maximum=200),
                before_updated_at=str(frame.get("before_updated_at") or "") or None,
                before_chat_id=str(frame.get("before_chat_id") or "") or None,
            )
            include_preview = bool(frame.get("include_preview"))
            await self._send({
                "type": FRAME_CHAT_LIST_RESPONSE,
                "request_id": request_id,
                "chats": [chat.to_response(include_preview=include_preview) for chat in page.chats],
                "has_more": page.has_more,
            })
        except ChatStateUnavailable as exc:
            await self._send_error(FRAME_CHAT_LIST_ERROR, request_id, exc.code, exc.message)
        except Exception:
            await self._send_error(FRAME_CHAT_LIST_ERROR, request_id, "CHAT_STATE_UNAVAILABLE")

    async def handle_snapshot(self, frame: dict[str, Any]) -> None:
        request_id = _clean_id(str(frame.get("request_id") or ""))
        chat_id = _clean_id(str(frame.get("chat_id") or ""))
        if not request_id:
            return
        if self._disabled:
            await self._send_error(FRAME_CHAT_SNAPSHOT_ERROR, request_id, "CHAT_CURSORS_DISABLED", chat_id=chat_id)
            return
        if not chat_id:
            await self._send_error(FRAME_CHAT_SNAPSHOT_ERROR, request_id, "CHAT_SNAPSHOT_INVALID", chat_id=chat_id)
            return
        if not hasattr(self._logs, "snapshot"):
            await self._send_error(FRAME_CHAT_SNAPSHOT_ERROR, request_id, "CHAT_STATE_UNAVAILABLE", chat_id=chat_id)
            return
        try:
            page = await asyncio.to_thread(
                self._logs.snapshot,
                chat_id,
                after_seq=_parse_seq(frame.get("after_seq")),
                limit=_parse_limit(frame.get("limit"), default=500, maximum=1000),
            )
            await self._send({
                "type": FRAME_CHAT_SNAPSHOT_RESPONSE,
                "request_id": request_id,
                "chat_id": chat_id,
                "chat": page.chat.to_response(include_preview=True) if page.chat else None,
                "events": [
                    {
                        "seq": event.seq,
                        "occurred_at": _iso_time(event.created_at),
                        "event": {
                            "kind": event.kind,
                            "payload": event.payload,
                        },
                    }
                    for event in page.events
                ],
                "next_after_seq": page.next_after_seq,
                "has_more": page.has_more,
                "deleted": page.deleted,
            })
        except ResumeExpired as exc:
            await self._send({
                "type": FRAME_CHAT_SNAPSHOT_ERROR,
                "request_id": request_id,
                "chat_id": chat_id,
                "code": "CHAT_RESUME_EXPIRED",
                "message": "Snapshot cursor is older than retained events",
                "floor_seq": exc.floor_seq,
                "latest_seq": exc.latest_seq,
            })
        except (ChatStateUnavailable, InvalidChatStateEvent) as exc:
            code = getattr(exc, "code", "CHAT_STATE_UNAVAILABLE")
            message = getattr(exc, "message", "")
            await self._send_error(FRAME_CHAT_SNAPSHOT_ERROR, request_id, code, message, chat_id=chat_id)
        except Exception:
            await self._send_error(FRAME_CHAT_SNAPSHOT_ERROR, request_id, "CHAT_STATE_UNAVAILABLE", chat_id=chat_id)

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
        try:
            log = self._logs.get_or_create(chat_id)
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
                if hasattr(self._logs, "db_live_bytes"):
                    await asyncio.to_thread(self._logs.acknowledge, chat_id, client_id, event.seq)
                else:
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
        except (ChatStateUnavailable, InvalidChatStateEvent) as exc:
            await self._send({
                "type": FRAME_CHAT_RESUME_EXPIRED,
                "request_id": request_id,
                "chat_id": chat_id,
                "code": getattr(exc, "code", "CHAT_STATE_UNAVAILABLE"),
                "message": getattr(exc, "message", ""),
                "floor_seq": 0,
                "latest_seq": 0,
            })
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if self._tasks.get(request_id) is current:
                self._tasks.pop(request_id, None)

    async def _send_error(
        self,
        frame_type: str,
        request_id: str,
        code: str,
        message: str = "",
        *,
        chat_id: str = "",
    ) -> None:
        frame: dict[str, Any] = {
            "type": frame_type,
            "request_id": request_id,
            "code": code,
            "message": message or code,
        }
        if chat_id:
            frame["chat_id"] = chat_id
        await self._send(frame)


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


def _parse_limit(value: Any, *, default: int, maximum: int) -> int:
    try:
        return min(max(1, int(value)), maximum)
    except (TypeError, ValueError):
        return default


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _iso_time(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat().replace("+00:00", "Z")
