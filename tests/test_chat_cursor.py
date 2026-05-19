from __future__ import annotations

import asyncio

import pytest

from hermes_vylen_gateway.chat_cursor import (
    FRAME_CHAT_EVENT,
    FRAME_CHAT_RESUME_EXPIRED,
    ChatCursorRelay,
)
from hermes_vylen_gateway.event_log import EventLogRegistry


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
    assert sent == [{"type": "push", "chat_id": "chat_a", "text": "kept", "seq": 1}]
    assert [event.seq for event in logs.get("chat_a").events] == [1]
    assert logs.get("chat_a").events[0].payload["seq"] == 1
