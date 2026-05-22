from __future__ import annotations

import asyncio

import pytest

from hermes_vylen_gateway.chat_cursor import (
    FRAME_CHAT_EVENT,
    FRAME_CHAT_SNAPSHOT_ERROR,
    FRAME_CHAT_LIST_RESPONSE,
    FRAME_CHAT_RESUME_EXPIRED,
    FRAME_CHAT_SNAPSHOT_RESPONSE,
    ChatCursorRelay,
)
from hermes_vylen_gateway.chat_store import ChatStateConfig, ChatStateStore
from hermes_vylen_gateway.event_log import EventLogRegistry


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_chat_subscribe_replays_after_client_cursor_and_tracks_per_client():
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    logs = EventLogRegistry()
    relay = ChatCursorRelay(send, logs)
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "one"})
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "two"})

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 1,
    })
    await asyncio.sleep(0)

    assert [frame["seq"] for frame in sent if frame["type"] == FRAME_CHAT_EVENT] == [2]
    assert logs.cursor("chat_a", "client_phone") == 2
    assert logs.cursor("chat_a", "client_tablet") == 0

    await relay.close()


@pytest.mark.asyncio
async def test_chat_event_replay_uses_retained_event_time():
    clock = Clock()
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    relay = ChatCursorRelay(send, EventLogRegistry(now=clock))
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "one"})
    clock.advance(10)
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "two"})

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 1,
    })
    await asyncio.sleep(0)

    events = [frame for frame in sent if frame["type"] == FRAME_CHAT_EVENT]
    assert events[0]["occurred_at"] == "1970-01-01T00:16:50Z"

    await relay.close()


@pytest.mark.asyncio
async def test_two_chat_subscribers_receive_live_push_without_consuming_each_other():
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    relay = ChatCursorRelay(send, EventLogRegistry())
    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_tablet",
        "chat_id": "chat_a",
        "client_id": "client_tablet",
        "after_seq": 0,
    })
    await asyncio.sleep(0)

    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "one"})
    await asyncio.sleep(0)

    by_request = {
        frame["request_id"]: frame["seq"]
        for frame in sent
        if frame["type"] == FRAME_CHAT_EVENT
    }
    assert by_request == {"req_phone": 1, "req_tablet": 1}

    await relay.close()


@pytest.mark.asyncio
async def test_chat_subscribe_reports_expired_cursor_when_retention_floor_advanced():
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    logs = EventLogRegistry(max_events=1)
    relay = ChatCursorRelay(send, logs)
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "one"})
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "two"})

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await asyncio.sleep(0)

    expired = [frame for frame in sent if frame["type"] == FRAME_CHAT_RESUME_EXPIRED]
    assert expired == [{
        "type": FRAME_CHAT_RESUME_EXPIRED,
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "code": "CHAT_RESUME_EXPIRED",
        "floor_seq": 1,
        "latest_seq": 2,
    }]


@pytest.mark.asyncio
async def test_same_request_resubscribe_old_task_does_not_pop_new_task():
    first_send_started = asyncio.Event()
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)
        if frame.get("chat_id") == "chat_a" and frame.get("type") == FRAME_CHAT_EVENT:
            first_send_started.set()
            await asyncio.Event().wait()

    relay = ChatCursorRelay(send, EventLogRegistry())
    relay.append_push({"type": "push", "chat_id": "chat_a", "text": "old"})
    relay.append_push({"type": "push", "chat_id": "chat_b", "text": "new"})

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_reused",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await asyncio.wait_for(first_send_started.wait(), timeout=1.0)

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_reused",
        "chat_id": "chat_b",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await asyncio.sleep(0)

    assert "req_reused" in relay._tasks
    assert any(
        frame.get("type") == FRAME_CHAT_EVENT and frame.get("chat_id") == "chat_b"
        for frame in sent
    )

    await relay.close()


@pytest.mark.asyncio
async def test_send_push_retains_only_after_gateway_send_succeeds():
    sent: list[dict] = []
    fail = True

    async def send(frame: dict) -> None:
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("socket closed")
        sent.append(dict(frame))

    logs = EventLogRegistry()
    relay = ChatCursorRelay(send, logs)

    with pytest.raises(RuntimeError, match="socket closed"):
        await relay.send_push({"type": "push", "chat_id": "chat_a", "text": "lost"})
    assert logs.get("chat_a") is None

    seq = await relay.send_push({"type": "push", "chat_id": "chat_a", "text": "kept"})

    assert seq == 1
    assert len(sent) == 1
    assert sent[0] == {
        "type": "push",
        "chat_id": "chat_a",
        "text": "kept",
        "seq": 1,
        "event_id": sent[0]["event_id"],
    }
    assert sent[0]["event_id"].startswith("evt_")
    assert [event.seq for event in logs.get("chat_a").events] == [1]
    assert logs.get("chat_a").events[0].payload["seq"] == 1
    assert logs.get("chat_a").events[0].payload["event_id"] == sent[0]["event_id"]


def test_append_push_adds_restart_stable_event_id_to_retained_payload():
    relay = ChatCursorRelay(lambda _: None, EventLogRegistry())

    seq = relay.append_push({"type": "push", "chat_id": "chat_a", "text": "kept"})

    assert seq == 1
    event = relay._logs.get("chat_a").events[0]
    assert event.payload["event_id"].startswith("evt_")


@pytest.mark.asyncio
async def test_chat_subscribe_replays_generic_retained_events():
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    logs = EventLogRegistry()
    relay = ChatCursorRelay(send, logs)

    seq = relay.append_event("chat_a", "message.created", {
        "message_id": "msg_1",
        "role": "user",
        "text": "hello",
    })

    assert seq == 1

    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "chat_a",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await asyncio.sleep(0)

    events = [frame for frame in sent if frame["type"] == FRAME_CHAT_EVENT]
    assert events[0]["event"] == {
        "kind": "message.created",
        "payload": {
            "message_id": "msg_1",
            "role": "user",
            "text": "hello",
        },
    }

    await relay.close()


@pytest.mark.asyncio
async def test_chat_list_and_snapshot_frames_read_sqlite_store(tmp_path):
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    store = ChatStateStore(tmp_path / "chat-state.sqlite3")
    relay = ChatCursorRelay(send, store)
    relay.append_event("chat_a", "message.created", {
        "message_id": "msg_1",
        "role": "user",
        "text": "hello",
    })

    await relay.handle_list({
        "type": "chat_list",
        "request_id": "req_list",
        "limit": 50,
        "include_preview": False,
    })
    await relay.handle_snapshot({
        "type": "chat_snapshot",
        "request_id": "req_snapshot",
        "chat_id": "chat_a",
        "after_seq": 0,
        "limit": 500,
    })

    list_response = next(frame for frame in sent if frame["type"] == FRAME_CHAT_LIST_RESPONSE)
    assert list_response["chats"][0]["chat_id"] == "chat_a"
    assert "preview" not in list_response["chats"][0]

    snapshot_response = next(frame for frame in sent if frame["type"] == FRAME_CHAT_SNAPSHOT_RESPONSE)
    assert snapshot_response["chat"]["chat_id"] == "chat_a"
    assert snapshot_response["events"][0]["event"]["payload"]["text"] == "hello"
    assert snapshot_response["next_after_seq"] == 1
    assert snapshot_response["has_more"] is False


@pytest.mark.asyncio
async def test_chat_snapshot_expired_cursor_uses_snapshot_error_frame(tmp_path):
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(frame)

    store = ChatStateStore(tmp_path / "chat-state.sqlite3", config=ChatStateConfig(max_events_per_chat=1))
    relay = ChatCursorRelay(send, store)
    relay.append_event("chat_a", "message.created", {"text": "one"})
    relay.append_event("chat_a", "message.created", {"text": "two"})
    store.sweep()

    await relay.handle_snapshot({
        "type": "chat_snapshot",
        "request_id": "req_snapshot",
        "chat_id": "chat_a",
        "after_seq": 0,
        "limit": 500,
    })

    error = next(frame for frame in sent if frame["type"] == FRAME_CHAT_SNAPSHOT_ERROR)
    assert error["code"] == "CHAT_RESUME_EXPIRED"
    assert error["floor_seq"] == 1


@pytest.mark.asyncio
async def test_active_quiet_subscription_keeps_log_attached_through_sweep():
    clock = Clock()
    logs = EventLogRegistry(ttl_seconds=1, now=clock)
    sent: list[dict] = []

    async def send(frame: dict) -> None:
        sent.append(dict(frame))

    relay = ChatCursorRelay(send, logs)
    await relay.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "req_phone",
        "chat_id": "inbox",
        "client_id": "client_phone",
        "after_seq": 0,
    })
    await asyncio.sleep(0)

    log = logs.get("inbox")
    assert log is not None
    assert log.active_tailers == 1

    clock.advance(2)
    assert logs.sweep() == 0
    assert logs.get("inbox") is log

    relay.append_push({"type": "push", "chat_id": "inbox", "text": "still live"})
    await asyncio.sleep(0)

    assert [
        frame["event"]["payload"]["text"]
        for frame in sent
        if frame.get("type") == FRAME_CHAT_EVENT
    ] == ["still live"]

    await relay.close()
