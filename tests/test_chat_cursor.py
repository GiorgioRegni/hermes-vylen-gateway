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
