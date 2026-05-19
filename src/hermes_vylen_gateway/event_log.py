"""Bounded in-memory event logs for resumable chat/run delivery."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

DEFAULT_CHAT_CURSOR_TTL_SECONDS = 900.0
DEFAULT_CHAT_CURSOR_MAX_EVENTS = 1000
DEFAULT_CHAT_CURSOR_MAX_BYTES = 4 * 1024 * 1024


class ResumeExpired(Exception):
    def __init__(self, floor_seq: int, latest_seq: int) -> None:
        super().__init__("resume cursor is older than retained events")
        self.floor_seq = floor_seq
        self.latest_seq = latest_seq


@dataclass(frozen=True)
class RetainedEvent:
    seq: int
    kind: str
    payload: Any
    created_at: float
    size_bytes: int


class RetainedEventLog:
    def __init__(
        self,
        key: str,
        *,
        ttl_seconds: float = DEFAULT_CHAT_CURSOR_TTL_SECONDS,
        max_events: int = DEFAULT_CHAT_CURSOR_MAX_EVENTS,
        max_bytes: int = DEFAULT_CHAT_CURSOR_MAX_BYTES,
        now: Any = time.time,
    ) -> None:
        self.key = key
        self.ttl_seconds = ttl_seconds
        self.max_events = max(1, max_events)
        self.max_bytes = max(1, max_bytes)
        self.created_at = float(now())
        self.updated_at = self.created_at
        self._now = now
        self._events: deque[RetainedEvent] = deque()
        self._next_seq = 1
        self._total_bytes = 0
        self._closed = False
        self._progressed = asyncio.Event()
        self._version = 0
        self._tailers = 0

    @property
    def events(self) -> tuple[RetainedEvent, ...]:
        return tuple(self._events)

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def latest_seq(self) -> int:
        return self._next_seq - 1

    @property
    def floor_seq(self) -> int:
        if self._events:
            return self._events[0].seq - 1
        return self.latest_seq

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_tailers(self) -> int:
        return self._tailers

    def append(self, kind: str, payload: Any, *, now: float | None = None) -> RetainedEvent:
        ts = float(self._now() if now is None else now)
        event = RetainedEvent(
            seq=self._next_seq,
            kind=kind,
            payload=payload,
            created_at=ts,
            size_bytes=_payload_size(kind, payload),
        )
        self._next_seq += 1
        self.updated_at = ts
        self._events.append(event)
        self._total_bytes += event.size_bytes
        self._evict(now=ts)
        self._bump()
        return event

    def close(self) -> None:
        self._closed = True
        self.updated_at = float(self._now())
        self._bump()

    def replay_after(self, after_seq: int) -> list[RetainedEvent]:
        self._evict()
        if after_seq < self.floor_seq:
            raise ResumeExpired(self.floor_seq, self.latest_seq)
        return [event for event in self._events if event.seq > after_seq]

    async def tail_after(
        self,
        after_seq: int,
        *,
        keepalive_seconds: float | None = None,
        reject_future_cursor: bool = False,
    ) -> AsyncIterator[RetainedEvent]:
        self._tailers += 1
        try:
            if reject_future_cursor and after_seq > self.latest_seq:
                raise ResumeExpired(self.floor_seq, self.latest_seq)
            cursor = after_seq
            while True:
                version = self._version
                for event in self.replay_after(cursor):
                    cursor = event.seq
                    yield event
                if self._closed and cursor >= self.latest_seq:
                    return
                self._progressed.clear()
                if self._version == version:
                    try:
                        if keepalive_seconds is None:
                            await self._progressed.wait()
                        else:
                            await asyncio.wait_for(self._progressed.wait(), timeout=keepalive_seconds)
                    except asyncio.TimeoutError:
                        continue
        finally:
            self._tailers -= 1

    def _evict(self, *, now: float | None = None) -> None:
        ts = float(self._now() if now is None else now)
        if self.ttl_seconds > 0:
            while self._events and ts - self._events[0].created_at > self.ttl_seconds:
                self._drop_oldest()
        while len(self._events) > self.max_events:
            self._drop_oldest()
        while self._events and self._total_bytes > self.max_bytes:
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        dropped = self._events.popleft()
        self._total_bytes -= dropped.size_bytes

    def _bump(self) -> None:
        self._version += 1
        self._progressed.set()


class EventLogRegistry:
    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CHAT_CURSOR_TTL_SECONDS,
        max_events: int = DEFAULT_CHAT_CURSOR_MAX_EVENTS,
        max_bytes: int = DEFAULT_CHAT_CURSOR_MAX_BYTES,
        now: Any = time.time,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._now = now
        self._logs: dict[str, RetainedEventLog] = {}
        self._cursors: dict[tuple[str, str], int] = {}

    def get_or_create(self, key: str) -> RetainedEventLog:
        log = self._logs.get(key)
        if log is None:
            log = RetainedEventLog(
                key,
                ttl_seconds=self.ttl_seconds,
                max_events=self.max_events,
                max_bytes=self.max_bytes,
                now=self._now,
            )
            self._logs[key] = log
        return log

    def get(self, key: str) -> RetainedEventLog | None:
        return self._logs.get(key)

    def drop(self, key: str) -> None:
        self._logs.pop(key, None)
        for cursor_key in list(self._cursors):
            if cursor_key[0] == key:
                self._cursors.pop(cursor_key, None)

    def acknowledge(self, key: str, client_id: str, seq: int) -> None:
        cursor_key = (key, client_id)
        self._cursors[cursor_key] = max(seq, self._cursors.get(cursor_key, 0))

    def cursor(self, key: str, client_id: str) -> int:
        return self._cursors.get((key, client_id), 0)

    def sweep(self, *, now: float | None = None) -> int:
        ts = float(self._now() if now is None else now)
        dropped: list[str] = []
        for key, log in self._logs.items():
            log._evict(now=ts)
            if log.active_tailers > 0:
                continue
            inactive_for = ts - log.updated_at
            if not log.events and inactive_for > self.ttl_seconds:
                dropped.append(key)
        for key in dropped:
            self.drop(key)
        return len(dropped)

    def __len__(self) -> int:
        return len(self._logs)


def _payload_size(kind: str, payload: Any) -> int:
    if isinstance(payload, bytes):
        return len(payload) + len(kind)
    try:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError):
        encoded = str(payload).encode("utf-8", errors="replace")
    return len(encoded) + len(kind)
