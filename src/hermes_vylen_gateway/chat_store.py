"""SQLite-backed shared chat state for Vylen chat replay."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from typing import Any, AsyncIterator

from .event_log import EventTooLarge, ResumeExpired, RetainedEvent

SCHEMA_VERSION = 1
DEFAULT_MAX_CHATS = 200
DEFAULT_MAX_EVENTS_PER_CHAT = 2000
DEFAULT_MAX_EVENT_BYTES = 1024 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_EVENT_TTL_DAYS = 30
DEFAULT_DELETED_TTL_DAYS = 7
DEFAULT_DEDUP_TTL_SECONDS = 900
DEFAULT_VACUUM_MIN_FREELIST_PAGES = 1024
DEFAULT_GC_APPEND_INTERVAL = 100
DEFAULT_GC_BYTES_INTERVAL = 8 * 1024 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_KIND_RE = re.compile(r"^(push|message|turn|activity|approval|confirm|session|chat)\.[A-Za-z0-9_.:-]+$|^push$")


class ChatStateUnavailable(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidChatStateEvent(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ChatStateConfig:
    max_chats: int = DEFAULT_MAX_CHATS
    max_events_per_chat: int = DEFAULT_MAX_EVENTS_PER_CHAT
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES
    max_bytes: int = DEFAULT_MAX_BYTES
    event_ttl_days: int = DEFAULT_EVENT_TTL_DAYS
    deleted_ttl_days: int = DEFAULT_DELETED_TTL_DAYS
    dedup_ttl_seconds: int = DEFAULT_DEDUP_TTL_SECONDS
    vacuum_min_freelist_pages: int = DEFAULT_VACUUM_MIN_FREELIST_PAGES
    gc_append_interval: int = DEFAULT_GC_APPEND_INTERVAL
    gc_bytes_interval: int = DEFAULT_GC_BYTES_INTERVAL


@dataclass(frozen=True)
class ChatRow:
    chat_id: str
    title: str
    kind: str
    created_at: str
    updated_at: str
    latest_seq: int
    floor_seq: int
    deleted_at: str | None = None
    last_message_preview: str | None = None
    last_message_role: str | None = None
    last_message_at: str | None = None

    def to_response(self, *, include_preview: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "chat_id": self.chat_id,
            "title": self.title,
            "kind": self.kind,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_seq": self.latest_seq,
            "floor_seq": self.floor_seq,
        }
        if include_preview:
            row["preview"] = {
                "last_message_preview": self.last_message_preview,
                "last_message_role": self.last_message_role,
                "last_message_at": self.last_message_at,
            }
        if self.deleted_at:
            row["deleted_at"] = self.deleted_at
        return row


@dataclass(frozen=True)
class ChatListPage:
    chats: list[ChatRow]
    has_more: bool


@dataclass(frozen=True)
class ChatSnapshotPage:
    chat: ChatRow | None
    events: list[RetainedEvent]
    next_after_seq: int
    has_more: bool
    deleted: bool = False


@dataclass(frozen=True)
class ChatStateStatus:
    status: str
    message: str | None = None
    quarantined_path: str | None = None


class ChatStateStore:
    """SQLite implementation with the EventLogRegistry surface used by cursors."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        config: ChatStateConfig | None = None,
        now: Any = time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.config = config or ChatStateConfig()
        self._now = now
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._logs: dict[str, ChatStateLog] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._status = ChatStateStatus(status="initializing")
        self._unavailable: ChatStateUnavailable | None = None
        self._appends_since_sweep = 0
        self._bytes_since_sweep = 0
        self._sweep_requested = False
        self._open()

    @classmethod
    def from_env(cls) -> "ChatStateStore":
        return cls(
            _chat_state_path_from_env(),
            config=ChatStateConfig(
                max_chats=_env_int("VYLEN_CHAT_STATE_MAX_CHATS", DEFAULT_MAX_CHATS),
                max_events_per_chat=_env_int("VYLEN_CHAT_STATE_MAX_EVENTS_PER_CHAT", DEFAULT_MAX_EVENTS_PER_CHAT),
                max_event_bytes=_env_bytes("VYLEN_CHAT_STATE_MAX_EVENT_BYTES", DEFAULT_MAX_EVENT_BYTES),
                max_bytes=_env_bytes("VYLEN_CHAT_STATE_MAX_BYTES", DEFAULT_MAX_BYTES),
                event_ttl_days=_env_int("VYLEN_CHAT_STATE_EVENT_TTL_DAYS", DEFAULT_EVENT_TTL_DAYS),
                deleted_ttl_days=_env_int("VYLEN_CHAT_STATE_DELETED_TTL_DAYS", DEFAULT_DELETED_TTL_DAYS),
                dedup_ttl_seconds=_env_int("VYLEN_CHAT_STATE_DEDUP_TTL_SECONDS", DEFAULT_DEDUP_TTL_SECONDS),
                vacuum_min_freelist_pages=_env_int(
                    "VYLEN_CHAT_STATE_VACUUM_MIN_FREELIST_PAGES",
                    DEFAULT_VACUUM_MIN_FREELIST_PAGES,
                ),
                gc_append_interval=_env_int("VYLEN_CHAT_STATE_GC_APPEND_INTERVAL", DEFAULT_GC_APPEND_INTERVAL),
                gc_bytes_interval=_env_bytes("VYLEN_CHAT_STATE_GC_BYTES_INTERVAL", DEFAULT_GC_BYTES_INTERVAL),
            ),
        )

    @property
    def status(self) -> ChatStateStatus:
        return self._status

    @property
    def unavailable(self) -> ChatStateUnavailable | None:
        return self._unavailable

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        with self._lock:
            self._loop = loop

    def get_or_create(self, key: str) -> "ChatStateLog":
        clean = _clean_id(key)
        if not clean:
            raise InvalidChatStateEvent("invalid_chat_id", "chat_id is invalid")
        with self._lock:
            log = self._logs.get(clean)
            if log is None:
                log = ChatStateLog(self, clean)
                self._logs[clean] = log
            return log

    def get(self, key: str) -> "ChatStateLog | None":
        clean = _clean_id(key)
        if not clean:
            return None
        if self.chat_exists(clean):
            return self.get_or_create(clean)
        return None

    def drop(self, key: str) -> None:
        clean = _clean_id(key)
        if not clean:
            return
        with self._lock:
            self._conn_or_raise().execute("DELETE FROM chats WHERE chat_id = ?", (clean,))
            self._conn_or_raise().commit()
            self._logs.pop(clean, None)

    def consume_sweep_requested(self) -> bool:
        with self._lock:
            if not self._sweep_requested:
                return False
            self._sweep_requested = False
            return True

    def acknowledge(self, key: str, client_id: str, seq: int) -> None:
        clean_chat_id = _clean_id(key)
        clean_client_id = _clean_id(client_id)
        if not clean_chat_id or not clean_client_id:
            return
        now = _iso_from_epoch(self._now())
        with self._lock:
            self._conn_or_raise().execute(
                """
                INSERT INTO client_cursors(chat_id, client_id, seq, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, client_id) DO UPDATE SET
                  seq = max(client_cursors.seq, excluded.seq),
                  updated_at = excluded.updated_at
                """,
                (clean_chat_id, clean_client_id, max(0, int(seq)), now),
            )
            self._conn_or_raise().commit()

    def cursor(self, key: str, client_id: str) -> int:
        clean_chat_id = _clean_id(key)
        clean_client_id = _clean_id(client_id)
        if not clean_chat_id or not clean_client_id:
            return 0
        with self._lock:
            row = self._conn_or_raise().execute(
                "SELECT seq FROM client_cursors WHERE chat_id = ? AND client_id = ?",
                (clean_chat_id, clean_client_id),
            ).fetchone()
            return int(row["seq"]) if row else 0

    def dedup_lookup(self, chat_id: str, client_message_id: str) -> dict[str, Any] | None:
        clean_chat_id = _clean_id(chat_id)
        clean_message_id = _clean_id(client_message_id)
        if not clean_chat_id or not clean_message_id:
            return None
        now = _iso_from_epoch(self._now())
        with self._lock:
            row = self._conn_or_raise().execute(
                """
                SELECT turn_id, message_id, payload_json FROM inbound_dedup
                WHERE chat_id = ? AND client_message_id = ? AND expires_at >= ?
                """,
                (clean_chat_id, clean_message_id, now),
            ).fetchone()
        if not row:
            return None
        payload: dict[str, Any] = {}
        if row["payload_json"]:
            try:
                parsed = json.loads(str(row["payload_json"]))
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = {}
        payload.setdefault("turn_id", str(row["turn_id"]))
        payload.setdefault("user_message_id", str(row["message_id"]))
        return payload

    def dedup_record(
        self,
        chat_id: str,
        client_message_id: str,
        *,
        turn_id: str,
        message_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        clean_chat_id = _clean_id(chat_id)
        clean_client_message_id = _clean_id(client_message_id)
        if not clean_chat_id or not clean_client_message_id:
            return
        accepted_at = _iso_from_epoch(self._now())
        expires_at = _iso_from_epoch(self._now() + self.config.dedup_ttl_seconds)
        with self._lock:
            self._conn_or_raise().execute(
                """
                INSERT INTO chats(chat_id, title, kind, created_at, updated_at, latest_seq, floor_seq)
                VALUES (?, 'New conversation', 'chat', ?, ?, 0, 0)
                ON CONFLICT(chat_id) DO NOTHING
                """,
                (clean_chat_id, accepted_at, accepted_at),
            )
            self._conn_or_raise().execute(
                """
                INSERT INTO inbound_dedup(
                  chat_id, client_message_id, turn_id, message_id, accepted_at, expires_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, client_message_id) DO UPDATE SET
                  turn_id = excluded.turn_id,
                  message_id = excluded.message_id,
                  accepted_at = excluded.accepted_at,
                  expires_at = excluded.expires_at,
                  payload_json = excluded.payload_json
                """,
                (
                    clean_chat_id,
                    clean_client_message_id,
                    turn_id,
                    message_id,
                    accepted_at,
                    expires_at,
                    _canonical_json(payload or {}),
                ),
            )
            self._conn_or_raise().commit()

    def dedup_forget(self, chat_id: str, client_message_id: str) -> None:
        clean_chat_id = _clean_id(chat_id)
        clean_message_id = _clean_id(client_message_id)
        if not clean_chat_id or not clean_message_id:
            return
        with self._lock:
            self._conn_or_raise().execute(
                "DELETE FROM inbound_dedup WHERE chat_id = ? AND client_message_id = ?",
                (clean_chat_id, clean_message_id),
            )
            self._conn_or_raise().commit()

    def chat_exists(self, chat_id: str) -> bool:
        with self._lock:
            row = self._conn_or_raise().execute(
                "SELECT 1 FROM chats WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            return row is not None

    def append_event(
        self,
        chat_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        now: float | None = None,
        title: str | None = None,
        creator_client_id: str | None = None,
        creator_user_id: str | None = None,
    ) -> RetainedEvent:
        self._raise_if_unavailable()
        clean_chat_id = _clean_id(chat_id)
        if not clean_chat_id:
            raise InvalidChatStateEvent("invalid_chat_id", "chat_id is invalid")
        if not _KIND_RE.match(kind or ""):
            raise InvalidChatStateEvent("invalid_event_kind", "event kind is invalid")
        payload_json = _canonical_json(payload)
        size_bytes = len(kind.encode("utf-8")) + len(payload_json.encode("utf-8"))
        if size_bytes > self.config.max_event_bytes:
            raise EventTooLarge(size_bytes, self.config.max_event_bytes)
        ts = float(self._now() if now is None else now)
        iso = _iso_from_epoch(ts)
        preview = _preview_from_event(kind, payload, iso)
        with self._lock:
            conn = self._conn_or_raise()
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT latest_seq FROM chats WHERE chat_id = ?",
                    (clean_chat_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO chats(
                          chat_id, title, kind, creator_client_id, creator_user_id,
                          created_at, updated_at, latest_seq, floor_seq
                        )
                        VALUES (?, ?, 'chat', ?, ?, ?, ?, 0, 0)
                        """,
                        (
                            clean_chat_id,
                            _clean_title(title or payload.get("chat_name") or "New conversation"),
                            _optional_id(creator_client_id),
                            _optional_id(creator_user_id),
                            iso,
                            iso,
                        ),
                    )
                    latest_seq = 0
                else:
                    latest_seq = int(row["latest_seq"])
                seq = latest_seq + 1
                conn.execute(
                    """
                    INSERT INTO chat_events(chat_id, seq, kind, payload_json, payload_bytes, occurred_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (clean_chat_id, seq, kind, payload_json, size_bytes, iso, iso),
                )
                if preview is None:
                    conn.execute(
                        "UPDATE chats SET latest_seq = ?, updated_at = ? WHERE chat_id = ?",
                        (seq, iso, clean_chat_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE chats SET
                          latest_seq = ?,
                          updated_at = ?,
                          last_message_preview = ?,
                          last_message_role = ?,
                          last_message_at = ?
                        WHERE chat_id = ?
                        """,
                        (seq, iso, preview[0], preview[1], preview[2], clean_chat_id),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            self._appends_since_sweep += 1
            self._bytes_since_sweep += size_bytes
            if (
                self.config.gc_append_interval > 0
                and self._appends_since_sweep >= self.config.gc_append_interval
            ) or (
                self.config.gc_bytes_interval > 0
                and self._bytes_since_sweep >= self.config.gc_bytes_interval
            ):
                self._sweep_requested = True
        self.get_or_create(clean_chat_id)._bump()
        return RetainedEvent(seq=seq, kind=kind, payload=dict(payload), created_at=ts, size_bytes=size_bytes)

    def ensure_fits(self, kind: str, payload: dict[str, Any]) -> int:
        payload_json = _canonical_json(payload)
        size_bytes = len(str(kind).encode("utf-8")) + len(payload_json.encode("utf-8"))
        if size_bytes > self.config.max_event_bytes:
            raise EventTooLarge(size_bytes, self.config.max_event_bytes)
        return size_bytes

    def replay_after(self, chat_id: str, after_seq: int, *, limit: int | None = None) -> list[RetainedEvent]:
        clean_chat_id = _clean_id(chat_id)
        if not clean_chat_id:
            raise InvalidChatStateEvent("invalid_chat_id", "chat_id is invalid")
        with self._lock:
            chat = self.get_chat(clean_chat_id, include_deleted=True)
            if chat is None:
                return []
            if after_seq < chat.floor_seq:
                raise ResumeExpired(chat.floor_seq, chat.latest_seq)
            sql = """
                SELECT seq, kind, payload_json, payload_bytes, occurred_at
                FROM chat_events
                WHERE chat_id = ? AND seq > ?
                ORDER BY seq ASC
            """
            params: tuple[Any, ...] = (clean_chat_id, max(0, int(after_seq)))
            if limit is not None:
                sql += " LIMIT ?"
                params = (clean_chat_id, max(0, int(after_seq)), max(0, int(limit)))
            rows = self._conn_or_raise().execute(sql, params).fetchall()
        return [_event_from_row(row) for row in rows]

    def latest_seq(self, chat_id: str) -> int:
        chat = self.get_chat(chat_id, include_deleted=True)
        return chat.latest_seq if chat else 0

    def floor_seq(self, chat_id: str) -> int:
        chat = self.get_chat(chat_id, include_deleted=True)
        return chat.floor_seq if chat else 0

    def list_chats(
        self,
        *,
        limit: int = 50,
        before_updated_at: str | None = None,
        before_chat_id: str | None = None,
        query: str | None = None,
        include_deleted: bool = False,
    ) -> ChatListPage:
        self._raise_if_unavailable()
        safe_limit = min(max(1, int(limit)), 200)
        where = []
        params: list[Any] = []
        clean_query = _clean_search_query(query or "")
        if not include_deleted:
            where.append("deleted_at IS NULL")
            where.append("chat_id != 'inbox'")
        if clean_query:
            where.append("lower(title) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(clean_query.lower())}%")
        clean_before_chat_id = _clean_id(before_chat_id or "")
        if before_updated_at and clean_before_chat_id:
            where.append("(updated_at < ? OR (updated_at = ? AND chat_id < ?))")
            params.extend([before_updated_at, before_updated_at, clean_before_chat_id])
        elif before_updated_at:
            where.append("updated_at < ?")
            params.append(before_updated_at)
        sql = """
            SELECT chat_id, title, kind, created_at, updated_at, latest_seq, floor_seq,
                   deleted_at, last_message_preview, last_message_role, last_message_at
            FROM chats
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC, chat_id DESC LIMIT ?"
        params.append(safe_limit + 1)
        with self._lock:
            rows = self._conn_or_raise().execute(sql, tuple(params)).fetchall()
        chats = [_chat_from_row(row) for row in rows[:safe_limit]]
        return ChatListPage(chats=chats, has_more=len(rows) > safe_limit)

    def get_chat(self, chat_id: str, *, include_deleted: bool = False) -> ChatRow | None:
        clean_chat_id = _clean_id(chat_id)
        if not clean_chat_id:
            return None
        sql = """
            SELECT chat_id, title, kind, created_at, updated_at, latest_seq, floor_seq,
                   deleted_at, last_message_preview, last_message_role, last_message_at
            FROM chats
            WHERE chat_id = ?
        """
        params: tuple[Any, ...] = (clean_chat_id,)
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        with self._lock:
            row = self._conn_or_raise().execute(sql, params).fetchone()
        return _chat_from_row(row) if row else None

    def snapshot(self, chat_id: str, *, after_seq: int = 0, limit: int = 500) -> ChatSnapshotPage:
        self._raise_if_unavailable()
        clean_chat_id = _clean_id(chat_id)
        if not clean_chat_id:
            raise InvalidChatStateEvent("invalid_chat_id", "chat_id is invalid")
        chat = self.get_chat(clean_chat_id, include_deleted=True)
        if chat is None:
            return ChatSnapshotPage(chat=None, events=[], next_after_seq=max(0, int(after_seq)), has_more=False)
        events = self.replay_after(clean_chat_id, max(0, int(after_seq)), limit=min(max(1, int(limit)), 1000) + 1)
        page_events = events[: min(max(1, int(limit)), 1000)]
        next_after_seq = page_events[-1].seq if page_events else max(0, int(after_seq))
        return ChatSnapshotPage(
            chat=chat,
            events=page_events,
            next_after_seq=next_after_seq,
            has_more=len(events) > len(page_events),
            deleted=bool(chat.deleted_at),
        )

    def mark_deleted(self, chat_id: str, *, now: float | None = None) -> RetainedEvent:
        clean_chat_id = _clean_id(chat_id)
        if not clean_chat_id:
            raise InvalidChatStateEvent("invalid_chat_id", "chat_id is invalid")
        ts = float(self._now() if now is None else now)
        iso = _iso_from_epoch(ts)
        with self._lock:
            self._conn_or_raise().execute(
                """
                INSERT INTO chats(chat_id, title, kind, created_at, updated_at, latest_seq, floor_seq, deleted_at)
                VALUES (?, 'Deleted conversation', 'chat', ?, ?, 0, 0, ?)
                ON CONFLICT(chat_id) DO UPDATE SET deleted_at = excluded.deleted_at, updated_at = excluded.updated_at
                """,
                (clean_chat_id, iso, iso, iso),
            )
            self._conn_or_raise().commit()
        return self.append_event(clean_chat_id, "chat.deleted", {"chat_id": clean_chat_id, "deleted_at": iso}, now=ts)

    def sweep(self, *, now: float | None = None) -> int:
        ts = float(self._now() if now is None else now)
        now_iso = _iso_from_epoch(ts)
        event_cutoff = _iso_from_epoch(ts - self.config.event_ttl_days * 86400)
        deleted_cutoff = _iso_from_epoch(ts - self.config.deleted_ttl_days * 86400)
        dedup_cutoff = _iso_from_epoch(ts - self.config.dedup_ttl_seconds)
        deleted_rows = 0
        with self._lock:
            conn = self._conn_or_raise()
            conn.execute("BEGIN IMMEDIATE")
            try:
                deleted_rows += conn.execute(
                    "DELETE FROM inbound_dedup WHERE expires_at < ? OR accepted_at < ?",
                    (now_iso, dedup_cutoff),
                ).rowcount
                chats = conn.execute("SELECT chat_id FROM chats").fetchall()
                for row in chats:
                    chat_id = row["chat_id"]
                    old = conn.execute(
                        "SELECT max(seq) AS max_seq FROM chat_events WHERE chat_id = ? AND created_at < ?",
                        (chat_id, event_cutoff),
                    ).fetchone()
                    deleted_rows += conn.execute(
                        "DELETE FROM chat_events WHERE chat_id = ? AND created_at < ?",
                        (chat_id, event_cutoff),
                    ).rowcount
                    if old and old["max_seq"] is not None:
                        conn.execute(
                            "UPDATE chats SET floor_seq = max(floor_seq, ?) WHERE chat_id = ?",
                            (int(old["max_seq"]), chat_id),
                        )
                    overflow = conn.execute(
                        """
                        SELECT seq FROM chat_events
                        WHERE chat_id = ?
                        ORDER BY seq DESC
                        LIMIT 1 OFFSET ?
                        """,
                        (chat_id, self.config.max_events_per_chat - 1),
                    ).fetchone()
                    if overflow:
                        floor = int(overflow["seq"]) - 1
                        deleted_rows += conn.execute(
                            "DELETE FROM chat_events WHERE chat_id = ? AND seq <= ?",
                            (chat_id, floor),
                        ).rowcount
                        conn.execute(
                            "UPDATE chats SET floor_seq = max(floor_seq, ?) WHERE chat_id = ?",
                            (floor, chat_id),
                        )
                deleted_rows += conn.execute(
                    "DELETE FROM chats WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (deleted_cutoff,),
                ).rowcount
                keep_chat_ids = {
                    row["chat_id"]
                    for row in conn.execute(
                        """
                        SELECT chat_id FROM chats
                        WHERE deleted_at IS NULL AND chat_id != 'inbox'
                        ORDER BY updated_at DESC, chat_id DESC
                        LIMIT ?
                        """,
                        (self.config.max_chats,),
                    ).fetchall()
                }
                active_chat_ids = {
                    chat_id
                    for chat_id, log in self._logs.items()
                    if log.active_tailers > 0
                }
                over_chats = conn.execute(
                    """
                    SELECT chat_id FROM chats
                    WHERE deleted_at IS NULL AND chat_id != 'inbox'
                    ORDER BY updated_at ASC, chat_id ASC
                    """
                ).fetchall()
                for row in over_chats:
                    chat_id = row["chat_id"]
                    if chat_id in keep_chat_ids or chat_id in active_chat_ids:
                        continue
                    deleted_rows += self._tombstone_chat_for_gc(conn, chat_id, now_iso, reason="max_chats")
                    remaining = conn.execute(
                        "SELECT count(*) AS count FROM chats WHERE deleted_at IS NULL AND chat_id != 'inbox'"
                    ).fetchone()
                    if remaining and int(remaining["count"]) <= self.config.max_chats:
                        break
                if self.db_live_bytes() > self.config.max_bytes:
                    inactive = conn.execute(
                        """
                        SELECT chat_id FROM chats
                        WHERE deleted_at IS NULL AND chat_id != 'inbox'
                        ORDER BY updated_at ASC, chat_id ASC
                        """
                    ).fetchall()
                    for row in inactive:
                        chat_id = row["chat_id"]
                        if chat_id in active_chat_ids:
                            continue
                        deleted_rows += self._tombstone_chat_for_gc(conn, chat_id, now_iso, reason="max_bytes")
                        if self.db_live_bytes() <= self.config.max_bytes:
                            break
                if self.db_live_bytes() > self.config.max_bytes:
                    conn.execute(
                        "INSERT INTO maintenance_log(kind, message, detail_json, created_at) VALUES (?, ?, ?, ?)",
                        (
                            "gc_budget_exceeded",
                            "Chat state database remains above byte budget after GC.",
                            _canonical_json({"live_bytes": self.db_live_bytes(), "max_bytes": self.config.max_bytes}),
                            now_iso,
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            self._appends_since_sweep = 0
            self._bytes_since_sweep = 0
            self._sweep_requested = False
            if deleted_rows:
                self._run_incremental_vacuum_if_needed()
        for log in self._logs.values():
            log._bump()
        return deleted_rows

    def db_live_bytes(self) -> int:
        with self._lock:
            page_size = int(self._conn_or_raise().execute("PRAGMA page_size").fetchone()[0])
            page_count = int(self._conn_or_raise().execute("PRAGMA page_count").fetchone()[0])
            freelist_count = int(self._conn_or_raise().execute("PRAGMA freelist_count").fetchone()[0])
        return max(0, page_count - freelist_count) * page_size

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._unavailable = ChatStateUnavailable(
                "chat_state_unavailable",
                f"chat state path is not writable: {exc}",
            )
            self._status = ChatStateStatus(status="open_failed", message=self._unavailable.message)
            return
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        try:
            version = self._read_user_version_readonly()
            if version > SCHEMA_VERSION:
                self._unavailable = ChatStateUnavailable(
                    "chat_state_unavailable",
                    f"chat state schema version {version} is newer than this plugin understands",
                )
                self._status = ChatStateStatus(status="version_mismatch", message=self._unavailable.message)
                return
            self._conn = self._connect()
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                self._create_schema()
            result = self._conn.execute("PRAGMA quick_check").fetchone()
            if not result or str(result[0]).lower() != "ok":
                raise sqlite3.DatabaseError(f"quick_check failed: {result[0] if result else 'no result'}")
            self._status = ChatStateStatus(status="ok")
            if os.environ.get("VYLEN_CHAT_STATE_GC_ON_STARTUP", "1") != "0":
                self.sweep()
        except sqlite3.DatabaseError as exc:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if isinstance(exc, sqlite3.OperationalError) and not self.path.exists():
                self._unavailable = ChatStateUnavailable(
                    "chat_state_unavailable",
                    f"chat state database could not be opened: {exc}",
                )
                self._status = ChatStateStatus(status="open_failed", message=self._unavailable.message)
                return
            quarantined = self._quarantine()
            self._conn = self._connect()
            self._create_schema()
            self._record_maintenance(
                "reset_after_corruption",
                "Local Vylen chat state was reset after SQLite corruption.",
                {"error": str(exc), "quarantined_path": quarantined},
            )
            self._status = ChatStateStatus(
                status="reset_after_corruption",
                message="Local Vylen chat state was reset after SQLite corruption.",
                quarantined_path=quarantined,
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        return conn

    def _read_user_version_readonly(self) -> int:
        if not self.path.exists():
            return 0
        uri_path = self.path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()

    def _tombstone_chat_for_gc(self, conn: sqlite3.Connection, chat_id: str, deleted_at: str, *, reason: str) -> int:
        row = conn.execute(
            "SELECT latest_seq, deleted_at FROM chats WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row is None or row["deleted_at"]:
            return 0
        latest_seq = int(row["latest_seq"])
        tombstone_seq = latest_seq + 1
        conn.execute("DELETE FROM chat_events WHERE chat_id = ?", (chat_id,))
        payload_json = _canonical_json({"chat_id": chat_id, "deleted_at": deleted_at, "reason": reason})
        conn.execute(
            """
            INSERT INTO chat_events(chat_id, seq, kind, payload_json, payload_bytes, occurred_at, created_at)
            VALUES (?, ?, 'chat.deleted', ?, ?, ?, ?)
            """,
            (chat_id, tombstone_seq, payload_json, len(payload_json), deleted_at, deleted_at),
        )
        conn.execute(
            """
            UPDATE chats
            SET deleted_at = ?, updated_at = ?, latest_seq = ?, floor_seq = ?
            WHERE chat_id = ?
            """,
            (deleted_at, deleted_at, tombstone_seq, latest_seq, chat_id),
        )
        return 1

    def _create_schema(self) -> None:
        conn = self._conn_or_raise()
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE IF NOT EXISTS chats (
              chat_id TEXT PRIMARY KEY,
              title TEXT NOT NULL DEFAULT 'New conversation',
              kind TEXT NOT NULL DEFAULT 'chat',
              creator_client_id TEXT,
              creator_user_id TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              latest_seq INTEGER NOT NULL DEFAULT 0,
              floor_seq INTEGER NOT NULL DEFAULT 0,
              deleted_at TEXT,
              last_message_preview TEXT,
              last_message_role TEXT,
              last_message_at TEXT
            );
            CREATE INDEX IF NOT EXISTS chats_updated_idx
              ON chats(updated_at DESC) WHERE deleted_at IS NULL;
            CREATE TABLE IF NOT EXISTS chat_events (
              chat_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              payload_bytes INTEGER NOT NULL,
              occurred_at TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (chat_id, seq),
              FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS chat_events_created_idx ON chat_events(created_at);
            CREATE TABLE IF NOT EXISTS client_cursors (
              chat_id TEXT NOT NULL,
              client_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (chat_id, client_id),
              FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS inbound_dedup (
              chat_id TEXT NOT NULL,
              client_message_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              message_id TEXT NOT NULL,
              accepted_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              payload_json TEXT,
              PRIMARY KEY (chat_id, client_message_id),
              FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS maintenance_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind TEXT NOT NULL,
              message TEXT NOT NULL,
              detail_json TEXT,
              created_at TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            COMMIT;
            """
        )

    def _record_maintenance(self, kind: str, message: str, detail: dict[str, Any] | None = None) -> None:
        self._conn_or_raise().execute(
            "INSERT INTO maintenance_log(kind, message, detail_json, created_at) VALUES (?, ?, ?, ?)",
            (kind, message, _canonical_json(detail or {}), _iso_from_epoch(self._now())),
        )
        self._conn_or_raise().commit()

    def _quarantine(self) -> str | None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        moved: str | None = None
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self.path) + suffix)
            if not src.exists():
                continue
            dst = Path(str(src) + f".corrupt-{stamp}")
            shutil.move(str(src), str(dst))
            if suffix == "":
                moved = str(dst)
        return moved

    def _run_incremental_vacuum_if_needed(self) -> None:
        freelist = int(self._conn_or_raise().execute("PRAGMA freelist_count").fetchone()[0])
        if freelist >= self.config.vacuum_min_freelist_pages:
            self._conn_or_raise().execute(f"PRAGMA incremental_vacuum({min(freelist, 256)})")

    def _conn_or_raise(self) -> sqlite3.Connection:
        self._raise_if_unavailable()
        if self._conn is None:
            raise ChatStateUnavailable("chat_state_unavailable", "chat state is not open")
        return self._conn

    def _raise_if_unavailable(self) -> None:
        if self._unavailable is not None:
            raise self._unavailable


class ChatStateLog:
    def __init__(self, store: ChatStateStore, chat_id: str) -> None:
        self._store = store
        self.key = chat_id
        self._progressed = asyncio.Event()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = store._loop
        self._closed = False
        self._version = 0
        self._tailers = 0

    @property
    def events(self) -> tuple[RetainedEvent, ...]:
        return tuple(self._store.replay_after(self.key, self.floor_seq))

    @property
    def next_seq(self) -> int:
        return self.latest_seq + 1

    @property
    def latest_seq(self) -> int:
        return self._store.latest_seq(self.key)

    @property
    def floor_seq(self) -> int:
        return self._store.floor_seq(self.key)

    @property
    def total_bytes(self) -> int:
        events = self.events
        return sum(event.size_bytes for event in events)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_tailers(self) -> int:
        return self._tailers

    def append(self, kind: str, payload: Any, *, now: float | None = None) -> RetainedEvent:
        event = self._store.append_event(self.key, kind, dict(payload), now=now)
        return event

    def ensure_fits(self, kind: str, payload: Any) -> int:
        return self._store.ensure_fits(kind, dict(payload))

    def close(self) -> None:
        self._closed = True
        self._bump()

    def replay_after(self, after_seq: int) -> list[RetainedEvent]:
        return self._store.replay_after(self.key, after_seq)

    async def tail_after(
        self,
        after_seq: int,
        *,
        keepalive_seconds: float | None = None,
        reject_future_cursor: bool = False,
    ) -> AsyncIterator[RetainedEvent]:
        self._tailers += 1
        try:
            if reject_future_cursor:
                latest_seq, floor_seq = await asyncio.to_thread(lambda: (self.latest_seq, self.floor_seq))
                if after_seq > latest_seq:
                    raise ResumeExpired(floor_seq, latest_seq)
            cursor = after_seq
            while True:
                version = self._version
                events = await asyncio.to_thread(self.replay_after, cursor)
                for event in events:
                    cursor = event.seq
                    yield event
                latest_seq = await asyncio.to_thread(lambda: self.latest_seq)
                if self._closed and cursor >= latest_seq:
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

    def _bump(self) -> None:
        self._version += 1
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if self._loop is not None and running_loop is not self._loop:
            self._loop.call_soon_threadsafe(self._progressed.set)
        else:
            self._progressed.set()


def _chat_state_path_from_env() -> str:
    override = os.environ.get("VYLEN_CHAT_STATE_DB_PATH")
    if override:
        return override
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return str(Path(hermes_home) / "vylen" / "chat-state.sqlite3")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_bytes(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = raw.strip().lower().replace(" ", "")
        multiplier = 1
        for suffix, factor in (("mib", 1024 * 1024), ("mb", 1024 * 1024), ("kib", 1024), ("kb", 1024)):
            if value.endswith(suffix):
                multiplier = factor
                value = value[: -len(suffix)]
                break
        return max(1, int(value) * multiplier)
    except ValueError:
        return default


def _clean_id(value: str) -> str:
    value = str(value or "").strip()
    if not _ID_RE.match(value):
        return ""
    return value


def _optional_id(value: str | None) -> str | None:
    if value is None:
        return None
    clean = _clean_id(value)
    return clean or None


def _clean_title(value: Any) -> str:
    text = str(value or "").strip() or "New conversation"
    return text[:200]


def _clean_search_query(value: Any) -> str:
    return str(value or "").strip()[:128]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _iso_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(float(epoch_seconds), timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch_from_iso(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _event_from_row(row: sqlite3.Row) -> RetainedEvent:
    return RetainedEvent(
        seq=int(row["seq"]),
        kind=str(row["kind"]),
        payload=json.loads(str(row["payload_json"])),
        created_at=_epoch_from_iso(str(row["occurred_at"])),
        size_bytes=int(row["payload_bytes"]),
    )


def _chat_from_row(row: sqlite3.Row) -> ChatRow:
    return ChatRow(
        chat_id=str(row["chat_id"]),
        title=str(row["title"]),
        kind=str(row["kind"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        latest_seq=int(row["latest_seq"]),
        floor_seq=int(row["floor_seq"]),
        deleted_at=row["deleted_at"],
        last_message_preview=row["last_message_preview"],
        last_message_role=row["last_message_role"],
        last_message_at=row["last_message_at"],
    )


def _preview_from_event(kind: str, payload: dict[str, Any], occurred_at: str) -> tuple[str, str, str] | None:
    if kind not in {"message.created", "message.updated"}:
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    role = str(payload.get("role") or "")
    return (text[:240], role, str(payload.get("created_at") or occurred_at))
