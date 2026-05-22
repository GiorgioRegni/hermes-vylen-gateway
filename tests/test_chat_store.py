from __future__ import annotations

import sqlite3

import pytest

from hermes_vylen_gateway.chat_store import (
    ChatStateConfig,
    ChatStateStore,
    ChatStateUnavailable,
    InvalidChatStateEvent,
)
from hermes_vylen_gateway.event_log import EventTooLarge, ResumeExpired


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_chat_store_creates_schema_and_replays_after_restart(tmp_path):
    path = tmp_path / "chat-state.sqlite3"
    first = ChatStateStore(path)
    event = first.append_event("chat_a", "message.created", {"text": "hello", "role": "user"})
    first.acknowledge("chat_a", "client_phone", event.seq)
    first.close()

    second = ChatStateStore(path)

    assert second.status.status == "ok"
    assert second.cursor("chat_a", "client_phone") == 1
    assert [event.payload["text"] for event in second.replay_after("chat_a", 0)] == ["hello"]
    assert second.get_chat("chat_a").latest_seq == 1


def test_inbound_dedup_survives_restart(tmp_path):
    path = tmp_path / "chat-state.sqlite3"
    first = ChatStateStore(path)
    first.append_event("chat_a", "message.created", {"text": "hello"})
    first.dedup_record(
        "chat_a",
        "client_msg_1",
        turn_id="turn_1",
        message_id="msg_user_1",
        payload={"turn_id": "turn_1", "user_message_id": "msg_user_1"},
    )
    first.close()

    second = ChatStateStore(path)

    assert second.dedup_lookup("chat_a", "client_msg_1") == {
        "turn_id": "turn_1",
        "user_message_id": "msg_user_1",
    }


def test_chat_store_rejects_invalid_ids_kind_and_oversized_events(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3", config=ChatStateConfig(max_event_bytes=32))

    with pytest.raises(InvalidChatStateEvent) as bad_id:
        store.append_event("../bad", "message.created", {"text": "hello"})
    assert bad_id.value.code == "invalid_chat_id"

    with pytest.raises(InvalidChatStateEvent) as bad_kind:
        store.append_event("chat_a", "bad", {"text": "hello"})
    assert bad_kind.value.code == "invalid_event_kind"

    with pytest.raises(EventTooLarge) as too_large:
        store.append_event("chat_a", "message.created", {"text": "x" * 200})
    assert too_large.value.max_bytes == 32
    assert store.get("chat_a") is None


def test_chat_list_defaults_to_no_preview_and_snapshot_paginates(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("chat_a", "message.created", {"text": "one", "role": "user"})
    store.append_event("chat_a", "message.created", {"text": "two", "role": "hermes"})
    store.append_event("chat_b", "message.created", {"text": "other", "role": "user"})

    page = store.list_chats(limit=10)
    assert [chat.chat_id for chat in page.chats] == ["chat_b", "chat_a"]
    assert "preview" not in page.chats[0].to_response()
    assert page.chats[1].to_response(include_preview=True)["preview"]["last_message_preview"] == "two"

    first = store.snapshot("chat_a", after_seq=0, limit=1)
    assert first.has_more is True
    assert first.next_after_seq == 1
    assert [event.seq for event in first.events] == [1]

    second = store.snapshot("chat_a", after_seq=first.next_after_seq, limit=1)
    assert second.has_more is False
    assert [event.seq for event in second.events] == [2]


def test_chat_list_keyset_pagination_keeps_same_timestamp_rows(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("chat_a", "message.created", {"text": "a"}, now=1000)
    store.append_event("chat_b", "message.created", {"text": "b"}, now=1000)

    first = store.list_chats(limit=1)
    second = store.list_chats(
        limit=1,
        before_updated_at=first.chats[-1].updated_at,
        before_chat_id=first.chats[-1].chat_id,
    )

    assert [chat.chat_id for chat in first.chats] == ["chat_b"]
    assert [chat.chat_id for chat in second.chats] == ["chat_a"]


def test_chat_list_searches_titles_with_keyset_pagination(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("chat_a", "message.created", {"text": "alpha"}, title="Deploy checklist", now=1000)
    store.append_event("chat_b", "message.created", {"text": "beta"}, title="deploy notes", now=1001)
    store.append_event("chat_c", "message.created", {"text": "deploy in content only"}, title="Unrelated", now=1002)

    first = store.list_chats(limit=1, query="DEPLOY")
    second = store.list_chats(
        limit=1,
        query="deploy",
        before_updated_at=first.chats[-1].updated_at,
        before_chat_id=first.chats[-1].chat_id,
    )

    assert [chat.chat_id for chat in first.chats] == ["chat_b"]
    assert first.has_more is True
    assert [chat.chat_id for chat in second.chats] == ["chat_a"]
    assert second.has_more is False


def test_chat_list_search_escapes_sqlite_wildcards(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("chat_literal", "message.created", {"text": "one"}, title="100%_literal", now=1000)
    store.append_event("chat_other", "message.created", {"text": "two"}, title="100xxliteral", now=1001)

    page = store.list_chats(limit=10, query="%_")

    assert [chat.chat_id for chat in page.chats] == ["chat_literal"]


def test_first_user_message_derives_plugin_title(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")

    store.append_event(
        "chat_a",
        "message.created",
        {"role": "user", "text": "Create a deploy checklist for Friday", "chat_name": "New conversation"},
        now=1000,
    )

    assert store.get_chat("chat_a").title == "Create a deploy checklist for Friday"


def test_rename_chat_persists_and_emits_retained_event(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("chat_a", "message.created", {"text": "hello", "role": "user"}, now=1000)

    event = store.rename_chat("chat_a", "Launch plan", now=1001)

    assert event.kind == "chat.renamed"
    assert event.payload["title"] == "Launch plan"
    assert store.get_chat("chat_a").title == "Launch plan"
    assert store.list_chats(query="launch").chats[0].chat_id == "chat_a"
    assert [event.kind for event in store.replay_after("chat_a", 0)] == ["message.created", "chat.renamed"]


def test_rename_chat_rejects_empty_title(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")

    with pytest.raises(InvalidChatStateEvent) as exc:
        store.rename_chat("chat_a", "  ")

    assert exc.value.code == "invalid_title"


def test_chat_list_excludes_plugin_inbox(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    store.append_event("inbox", "push", {"text": "cron"})
    store.append_event("chat_a", "message.created", {"text": "hello"})

    page = store.list_chats(limit=10)

    assert [chat.chat_id for chat in page.chats] == ["chat_a"]


def test_gc_advances_floor_and_expires_old_cursor(tmp_path):
    store = ChatStateStore(
        tmp_path / "chat-state.sqlite3",
        config=ChatStateConfig(max_events_per_chat=2, vacuum_min_freelist_pages=1),
    )
    for index in range(4):
        store.append_event("chat_a", "message.created", {"text": str(index)})

    store.sweep()

    chat = store.get_chat("chat_a")
    assert chat.floor_seq == 2
    assert [event.seq for event in store.replay_after("chat_a", 2)] == [3, 4]
    with pytest.raises(ResumeExpired) as expired:
        store.replay_after("chat_a", 1)
    assert expired.value.floor_seq == 2
    assert expired.value.latest_seq == 4


def test_gc_does_not_delete_active_chat_over_max_chat_cap(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3", config=ChatStateConfig(max_chats=1))
    store.append_event("chat_active", "message.created", {"text": "active"}, now=1000)
    store.append_event("chat_new", "message.created", {"text": "new"}, now=1001)
    store.get_or_create("chat_active")._tailers = 1

    store.sweep()

    assert store.get_chat("chat_active") is not None
    assert store.get_chat("chat_new") is not None


def test_gc_records_budget_exceeded_when_byte_cap_cannot_be_met(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3", config=ChatStateConfig(max_bytes=1))
    store.append_event("inbox", "message.created", {"text": "inbox"})

    store.sweep()

    rows = store._conn_or_raise().execute(
        "SELECT kind FROM maintenance_log WHERE kind = 'gc_budget_exceeded'"
    ).fetchall()
    assert len(rows) >= 1


def test_gc_deletes_old_deleted_tombstones(tmp_path):
    clock = Clock()
    store = ChatStateStore(
        tmp_path / "chat-state.sqlite3",
        config=ChatStateConfig(deleted_ttl_days=1),
        now=clock,
    )
    store.append_event("chat_a", "message.created", {"text": "hello"})
    store.mark_deleted("chat_a")
    assert store.get_chat("chat_a", include_deleted=True).deleted_at is not None

    clock.advance(2 * 86400)
    store.sweep()

    assert store.get_chat("chat_a", include_deleted=True) is None


def test_mark_deleted_unknown_chat_creates_hidden_tombstone(tmp_path):
    store = ChatStateStore(tmp_path / "chat-state.sqlite3")

    store.mark_deleted("chat_missing")

    assert store.get_chat("chat_missing") is None
    tombstone = store.get_chat("chat_missing", include_deleted=True)
    assert tombstone is not None
    assert tombstone.deleted_at is not None
    assert [event.kind for event in store.replay_after("chat_missing", 0)] == ["chat.deleted"]


def test_corrupt_database_is_quarantined_and_recreated(tmp_path):
    path = tmp_path / "chat-state.sqlite3"
    path.write_bytes(b"not sqlite")

    store = ChatStateStore(path)

    assert store.status.status == "reset_after_corruption"
    assert store.status.quarantined_path is not None
    assert path.exists()
    assert list(tmp_path.glob("chat-state.sqlite3.corrupt-*"))
    rows = store._conn_or_raise().execute("SELECT kind FROM maintenance_log").fetchall()
    assert [row["kind"] for row in rows] == ["reset_after_corruption"]


def test_newer_schema_enters_degraded_mode_without_quarantine(tmp_path):
    path = tmp_path / "chat-state.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 999")
    conn.close()

    store = ChatStateStore(path)

    assert store.status.status == "version_mismatch"
    assert not list(tmp_path.glob("chat-state.sqlite3.corrupt-*"))
    with pytest.raises(ChatStateUnavailable):
        store.list_chats()
