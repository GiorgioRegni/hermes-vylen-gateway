from __future__ import annotations

import base64
import asyncio
import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

import hermes_vylen_gateway.adapter as adapter_mod
from hermes_vylen_gateway.chat_cursor import FRAME_CHAT_EVENT, ChatCursorRelay
from hermes_vylen_gateway.chat_store import ChatStateUnavailable


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(dict(frame))


class FakePlatform:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class FakeSessionEntry:
    session_id: str


class FakeSessionStore:
    def __init__(self) -> None:
        self.appended: list[tuple[str, dict[str, Any]]] = []
        self._db: FakeSessionDB | None = None

    def get_or_create_session(self, source) -> FakeSessionEntry:
        return FakeSessionEntry(session_id=f"session_{source.chat_id}_{source.user_id}")

    def append_to_transcript(self, session_id: str, message: dict[str, Any]) -> None:
        self.appended.append((session_id, dict(message)))


class FakeSessionDB:
    def __init__(self) -> None:
        self.titles: list[tuple[str, str]] = []

    def sanitize_title(self, title: str) -> str:
        return " ".join(str(title).strip().split())[:100]

    def set_session_title(self, session_id: str, title: str) -> bool:
        self.titles.append((session_id, title))
        return True


class FakeSendResult:
    def __init__(self, success: bool, message_id: str | None = None, error: str | None = None, **kwargs) -> None:
        self.success = success
        self.message_id = message_id
        self.error = error


class FakeMessageType:
    TEXT = "text"
    PHOTO = "photo"
    AUDIO = "audio"
    VOICE = "voice"


@dataclass
class FakeSessionSource:
    platform: Any
    chat_id: str
    chat_name: str | None = None
    chat_type: str = "dm"
    user_id: str | None = None
    user_name: str | None = None
    message_id: str | None = None


@dataclass
class FakeMessageEvent:
    text: str
    message_type: Any = FakeMessageType.TEXT
    source: FakeSessionSource | None = None
    raw_message: Any = None
    message_id: str | None = None
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)


class FakeBasePlatformAdapter:
    def __init__(self, config, platform) -> None:
        self.config = config
        self.platform = platform
        self._message_handler = None
        self.handled_events: list[Any] = []
        self.cancelled_sessions: list[str] = []
        self.cancel_kwargs: list[dict[str, Any]] = []
        self._active_sessions: dict[str, Any] = {}
        self._pending_messages: dict[str, Any] = {}
        self.started_sessions: list[tuple[str, Any]] = []

    def set_message_handler(self, handler) -> None:
        self._message_handler = handler

    async def handle_message(self, event) -> None:
        self.handled_events.append(event)
        if self._message_handler is None:
            return
        if event.text == "/stop":
            return
        await self.on_processing_start(event)
        outcome = types.SimpleNamespace(value="success")
        try:
            response = await self._message_handler(event)
            if response:
                result = await self.send(event.source.chat_id, response, metadata={})
                if result.success and result.message_id:
                    await self.edit_message(
                        event.source.chat_id,
                        result.message_id,
                        response,
                        finalize=True,
                    )
        except Exception:
            outcome = types.SimpleNamespace(value="failure")
            raise
        finally:
            await self.on_processing_complete(event, outcome)

    async def cancel_session_processing(self, session_key, **kwargs) -> None:
        self.cancelled_sessions.append(session_key)
        self.cancel_kwargs.append(dict(kwargs))

    def _start_session_processing(self, event, session_key, **kwargs) -> bool:
        self.started_sessions.append((session_key, event))
        return True


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("VYLEN_CHAT_STATE_DB_PATH", str(tmp_path / "chat-state.sqlite3"))
    gateway_mod = types.ModuleType("gateway")
    platforms_mod = types.ModuleType("gateway.platforms")
    base_mod = types.ModuleType("gateway.platforms.base")
    base_mod.MessageEvent = FakeMessageEvent
    base_mod.MessageType = FakeMessageType
    base_mod.SendResult = FakeSendResult
    base_mod.cache_image_from_bytes = lambda data, ext=".jpg": f"/tmp/fake-image{ext}"
    base_mod.cache_audio_from_bytes = lambda data, ext=".ogg": f"/tmp/fake-audio{ext}"
    session_mod = types.ModuleType("gateway.session")
    session_mod.SessionSource = FakeSessionSource
    session_mod.build_session_key = lambda source, **kwargs: f"{source.platform.value}:{source.chat_id}:{source.user_id}"

    monkeypatch.setitem(sys.modules, "gateway", gateway_mod)
    monkeypatch.setitem(sys.modules, "gateway.platforms", platforms_mod)
    monkeypatch.setitem(sys.modules, "gateway.platforms.base", base_mod)
    monkeypatch.setitem(sys.modules, "gateway.session", session_mod)
    monkeypatch.setattr(
        adapter_mod,
        "_import_hermes",
        lambda: (FakeBasePlatformAdapter, FakePlatform),
    )

    cls = adapter_mod.make_adapter_class()
    instance = cls(config=types.SimpleNamespace(extra={}))
    client = FakeClient()
    instance._instance_id = "inst_1"
    instance._client = client
    instance._chat_event_logs.set_event_loop(asyncio.get_event_loop())
    instance._chat_cursors = ChatCursorRelay(client.send, instance._chat_event_logs)
    instance._fake_client = client
    return instance


def _chat_message_frame(**overrides):
    frame = {
        "type": "chat_message",
        "request_id": "req_1",
        "chat_id": "chat_a",
        "client_message_id": "client_msg_1",
        "client_id": "phone",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "chat_name": "Planning",
        "text": "hello",
    }
    frame.update(overrides)
    return frame


def test_authorize_vylen_user_merges_cloud_ready_user(monkeypatch):
    monkeypatch.setenv("VYLEN_ALLOWED_USERS", "existing")

    adapter_mod._authorize_vylen_user(" dev ")
    adapter_mod._authorize_vylen_user("dev")
    adapter_mod._authorize_vylen_user("")

    assert os.environ["VYLEN_ALLOWED_USERS"] == "existing,dev"


def test_parse_tool_progress_requires_emoji_prefixed_progress_lines():
    assert adapter_mod._parse_tool_progress('🔍 read_file: "Makefile"') == [{
        "emoji": "🔍",
        "tool": "read_file",
        "label": "Makefile",
    }]
    assert adapter_mod._parse_tool_progress("⚙️ shell...") == [{
        "emoji": "⚙️",
        "tool": "shell",
        "label": "",
    }]
    assert adapter_mod._parse_tool_progress("Note: not a tool") == []
    assert adapter_mod._parse_tool_progress("Received choice: **detailed**.") == []
    assert adapter_mod._parse_tool_progress("- read_file: Makefile") == []
    assert adapter_mod._parse_tool_progress("* shell...") == []


async def test_chat_event_append_emits_chat_index_changed_with_preview(adapter, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _run_sync_as_async)
    adapter._fake_client.sent.clear()

    await adapter._append_chat_event_async("chat_a", "message.created", {
        "message_id": "msg_1",
        "role": "user",
        "text": "hello from user",
        "created_at": "2026-05-23T12:00:00Z",
    })

    frames = [
        frame
        for frame in adapter._fake_client.sent
        if frame["type"] == adapter_mod.FRAME_CHAT_INDEX_CHANGED
    ]
    assert frames
    frame = frames[-1]
    assert frame["instance_id"] == "inst_1"
    assert frame["chat_id"] == "chat_a"
    assert frame["chat"]["chat_id"] == "chat_a"
    assert frame["chat"]["preview"]["last_message_preview"] == "hello from user"


async def test_chat_message_update_does_not_emit_chat_index_changed(adapter, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _run_sync_as_async)
    await adapter._append_chat_event_async("chat_a", "message.created", {
        "message_id": "msg_1",
        "role": "hermes",
        "text": "initial",
        "created_at": "2026-05-23T12:00:00Z",
    })
    adapter._fake_client.sent.clear()

    await adapter._append_chat_event_async("chat_a", "message.updated", {
        "message_id": "msg_1",
        "role": "hermes",
        "text": "streaming update",
        "updated_at": "2026-05-23T12:00:01Z",
    })

    assert not [
        frame
        for frame in adapter._fake_client.sent
        if frame["type"] == adapter_mod.FRAME_CHAT_INDEX_CHANGED
    ]


async def test_deleted_chat_emits_chat_index_changed_deleted_frame(adapter, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _run_sync_as_async)
    await adapter._append_chat_event_async("chat_a", "message.created", {
        "message_id": "msg_1",
        "role": "user",
        "text": "hello",
        "created_at": "2026-05-23T12:00:00Z",
    })
    adapter._fake_client.sent.clear()

    await asyncio.to_thread(adapter._chat_event_logs.mark_deleted, "chat_a")
    await adapter._emit_chat_index_changed("chat_a")

    frames = [
        frame
        for frame in adapter._fake_client.sent
        if frame["type"] == adapter_mod.FRAME_CHAT_INDEX_CHANGED
    ]
    assert frames
    assert frames[-1]["instance_id"] == "inst_1"
    assert frames[-1]["chat_id"] == "chat_a"
    assert frames[-1]["deleted"] is True
    assert frames[-1]["chat"]["chat_id"] == "chat_a"


async def test_chat_index_changed_skips_unavailable_chat_state(adapter, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _run_sync_as_async)

    class UnavailableChatState:
        def get_chat(self, *args, **kwargs):
            raise ChatStateUnavailable("chat state unavailable")

    adapter._fake_client.sent.clear()
    adapter._chat_event_logs = UnavailableChatState()

    await adapter._emit_chat_index_changed("chat_a")

    assert not [
        frame
        for frame in adapter._fake_client.sent
        if frame["type"] == adapter_mod.FRAME_CHAT_INDEX_CHANGED
    ]


async def _run_sync_as_async(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def test_tool_progress_activity_id_is_stable_when_label_changes():
    first = adapter_mod._activity_id("turn_1", "msg_progress", 0, {
        "tool": "read_file",
        "label": "Makefile",
    })
    second = adapter_mod._activity_id("turn_1", "msg_progress", 0, {
        "tool": "read_file",
        "label": "Makefile and docs/dev.md",
    })

    assert first == second


@pytest.mark.asyncio
async def test_chat_message_deduplicates_by_chat_and_client_message_id(adapter):
    handled: list[FakeMessageEvent] = []

    async def handler(event):
        handled.append(event)
        return "pong"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame())
    await adapter._handle_chat_message(_chat_message_frame(request_id="req_2"))

    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert len(acks) == 2
    assert acks[0]["turn_id"] == acks[1]["turn_id"]
    assert len(handled) == 1


@pytest.mark.asyncio
async def test_status_and_reset_with_args_are_regular_messages(adapter):
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        return "pong"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="/status report"))
    await adapter._handle_chat_message(_chat_message_frame(
        request_id="req_2",
        client_message_id="client_msg_2",
        text="/reset please summarize",
    ))

    assert handled == ["/status report", "/reset please summarize"]
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_error"]
    assert errors == []


@pytest.mark.asyncio
async def test_chat_message_during_active_run_queues_instead_of_interrupting(adapter):
    handled: list[FakeMessageEvent] = []

    async def handler(event):
        handled.append(event)
        return "pong"

    adapter.set_message_handler(handler)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_message(_chat_message_frame(text="follow up during run"))

    events = adapter._chat_event_logs.get("chat_a").events
    created = [event for event in events if event.kind == "message.created"][-1]
    queued = [event for event in events if event.kind == "turn.queued"]
    assert created.payload["text"] == "follow up during run"
    assert created.payload["status"] == "queued"
    assert queued
    assert handled == []
    assert adapter._pending_messages
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert len(acks) == 1


@pytest.mark.asyncio
async def test_message_queue_action_enqueues_followup(adapter):
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "queue_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_queue_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.queue",
        "text": "queued follow-up",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    created = [event for event in events if event.kind == "message.created"][-1]
    assert created.payload["client_message_id"] == "client_queue_1"
    assert created.payload["status"] == "queued"
    assert created.payload["intent"] == "queue"
    assert [event for event in events if event.kind == "turn.queued"]
    assert adapter._pending_messages
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert len(acks) == 1
    assert acks[0]["client_message_id"] == "client_queue_1"


@pytest.mark.asyncio
async def test_message_queue_action_is_idempotent(adapter):
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}
    frame = {
        "type": "chat_action",
        "request_id": "queue_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_queue_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.queue",
        "text": "queued follow-up",
    }

    await adapter._handle_chat_action(dict(frame))
    frame["request_id"] = "queue_req_retry"
    await adapter._handle_chat_action(dict(frame))

    events = adapter._chat_event_logs.get("chat_a").events
    created = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("client_message_id") == "client_queue_1"
    ]
    queued = [event for event in events if event.kind == "turn.queued"]
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert len(created) == 1
    assert len(queued) == 1
    assert len(acks) == 2
    assert acks[0]["message_id"] == acks[1]["message_id"]
    assert acks[1]["message_status"] == "queued"


@pytest.mark.asyncio
async def test_message_queue_action_preserves_fifo_without_runner(adapter):
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    for idx, text in enumerate(["first", "second", "third"], start=1):
        await adapter._handle_chat_action({
            "type": "chat_action",
            "request_id": f"queue_req_{idx}",
            "chat_id": "chat_a",
            "client_id": "phone",
            "client_message_id": f"client_queue_{idx}",
            "user_id": "user_1",
            "user_name": "Giorgio",
            "action": "message.queue",
            "text": text,
        })

    session_key = "vylen:chat_a:user_1"
    assert adapter._pending_messages[session_key].text == "first"
    assert [event.text for event in adapter._queued_events[session_key]] == ["second", "third"]

    first = adapter._pending_messages.pop(session_key)
    await adapter.on_processing_complete(first, types.SimpleNamespace(value="success"))
    assert adapter._pending_messages[session_key].text == "second"
    assert [event.text for event in adapter._queued_events[session_key]] == ["third"]


@pytest.mark.asyncio
async def test_runner_drained_queued_turn_gets_its_own_lifecycle_and_output(adapter):
    class FakeRunner:
        async def _handle_message(self, event):
            return None

        async def _run_agent(self, **kwargs):
            await adapter.send("chat_a", "💻 terminal: date")
            await adapter.send("chat_a", "Sun May 24 04:14:50 UTC 2026")
            return {"final_response": "Sun May 24 04:14:50 UTC 2026", "messages": []}

    runner = FakeRunner()
    adapter.set_message_handler(runner._handle_message)
    original_turn_id = "turn_waiting_for_clarify"
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": original_turn_id, "message_id": "msg_active"}
    queued_message_id = "msg_user_queued"
    queued_turn_id = "turn_queued_time"
    queued_event = adapter._build_message_event(
        frame={
            "request_id": "queue_req_1",
            "chat_id": "chat_a",
            "client_id": "phone",
            "client_message_id": "client_queue_1",
            "user_id": "user_1",
            "user_name": "Giorgio",
            "text": "what time is it?",
        },
        chat_id="chat_a",
        text="what time is it?",
        user_message_id=queued_message_id,
        turn_id=queued_turn_id,
        media_urls=[],
        media_types=[],
    )
    await adapter._append_chat_event_async("chat_a", "message.created", {
        "message_id": queued_message_id,
        "role": "user",
        "text": "what time is it?",
        "status": "queued",
        "created_at": "2026-05-24T04:14:00Z",
        "turn_id": queued_turn_id,
        "client_message_id": "client_queue_1",
    })
    adapter._queued_events_by_message_id[queued_message_id] = queued_event
    session_key = "vylen:chat_a:user_1"
    await asyncio.wait_for(
        runner._run_agent(
            source=queued_event.source,
            session_key=session_key,
            event_message_id=queued_message_id,
        ),
        timeout=5,
    )

    events = adapter._chat_event_logs.get("chat_a").events
    started = [event for event in events if event.kind == "turn.started"]
    completed = [event for event in events if event.kind == "turn.completed"]
    queued_user_updates = [
        event for event in events
        if event.kind == "message.updated" and event.payload.get("message_id") == queued_message_id
    ]
    activity_events = [event for event in events if event.kind == "activity.started"]
    assistant_messages = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    ]
    assistant_text_updates = [
        event for event in events
        if event.kind == "message.updated"
        and event.payload.get("turn_id") == queued_turn_id
        and event.payload.get("text") == "Sun May 24 04:14:50 UTC 2026"
    ]

    assert started[-1].payload["turn_id"] == queued_turn_id
    assert completed[-1].payload["turn_id"] == queued_turn_id
    assert queued_user_updates[0].payload["status"] == "running"
    assert queued_user_updates[-1].payload["status"] == "completed"
    assert activity_events[-1].payload["turn_id"] == queued_turn_id
    assert assistant_messages[-1].payload["turn_id"] == queued_turn_id
    assert assistant_text_updates
    assert adapter._active_turns_by_chat["chat_a"]["turn_id"] == original_turn_id


@pytest.mark.asyncio
async def test_runner_turn_lifecycle_wrapper_uses_latest_adapter_after_reconnect(adapter):
    class FakeRunner:
        def __init__(self) -> None:
            self.adapter = adapter

        async def _handle_message(self, event):
            return None

        async def _run_agent(self, **kwargs):
            await self.adapter.send("chat_a", "runner response after reconnect")
            return {"final_response": "runner response after reconnect", "messages": []}

    runner = FakeRunner()
    adapter.set_message_handler(runner._handle_message)

    second_adapter = type(adapter)(config=types.SimpleNamespace(extra={}))
    second_client = FakeClient()
    second_adapter._instance_id = "inst_2"
    second_adapter._client = second_client
    second_adapter._chat_event_logs.set_event_loop(asyncio.get_event_loop())
    second_adapter._chat_cursors = ChatCursorRelay(second_client.send, second_adapter._chat_event_logs)
    second_adapter._fake_client = second_client
    runner.adapter = second_adapter
    second_adapter.set_message_handler(runner._handle_message)

    queued_message_id = "msg_user_queued_reconnect"
    queued_turn_id = "turn_queued_reconnect"
    queued_event = second_adapter._build_message_event(
        frame={
            "request_id": "queue_req_reconnect",
            "chat_id": "chat_a",
            "client_id": "phone",
            "client_message_id": "client_queue_reconnect",
            "user_id": "user_1",
            "user_name": "Giorgio",
            "text": "queued after reconnect",
        },
        chat_id="chat_a",
        text="queued after reconnect",
        user_message_id=queued_message_id,
        turn_id=queued_turn_id,
        media_urls=[],
        media_types=[],
    )
    await second_adapter._append_chat_event_async("chat_a", "message.created", {
        "message_id": queued_message_id,
        "role": "user",
        "text": "queued after reconnect",
        "status": "queued",
        "created_at": "2026-05-24T04:14:00Z",
        "turn_id": queued_turn_id,
        "client_message_id": "client_queue_reconnect",
    })
    second_adapter._queued_events_by_message_id[queued_message_id] = queued_event

    await runner._run_agent(
        source=queued_event.source,
        session_key="vylen:chat_a:user_1",
        event_message_id=queued_message_id,
    )

    second_events = second_adapter._chat_event_logs.get("chat_a").events
    second_started = [event for event in second_events if event.kind == "turn.started"]
    second_completed = [event for event in second_events if event.kind == "turn.completed"]
    second_assistant = [
        event for event in second_events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    ]

    assert getattr(runner, "_vylen_turn_lifecycle_adapter") is second_adapter
    assert queued_message_id not in adapter._queued_events_by_message_id
    assert queued_message_id not in second_adapter._queued_events_by_message_id
    assert second_started[-1].payload["turn_id"] == queued_turn_id
    assert second_completed[-1].payload["turn_id"] == queued_turn_id
    assert second_assistant[-1].payload["turn_id"] == queued_turn_id


@pytest.mark.asyncio
async def test_normal_queued_turn_start_clears_runner_drain_bookkeeping(adapter):
    class FakeRunner:
        async def _handle_message(self, event):
            return None

        async def _run_agent(self, **kwargs):
            await adapter.send("chat_a", "should not be turn-bound")
            return {"final_response": "should not be turn-bound", "messages": []}

    runner = FakeRunner()
    adapter.set_message_handler(runner._handle_message)
    queued_event = adapter._build_message_event(
        frame={
            "request_id": "queue_req_1",
            "chat_id": "chat_a",
            "client_id": "phone",
            "client_message_id": "client_queue_1",
            "user_id": "user_1",
            "user_name": "Giorgio",
            "text": "normal drained queue",
        },
        chat_id="chat_a",
        text="normal drained queue",
        user_message_id="msg_user_queued",
        turn_id="turn_normal_queue",
        media_urls=[],
        media_types=[],
    )
    adapter._queued_events_by_message_id["msg_user_queued"] = queued_event

    await adapter.on_processing_start(queued_event)
    assert "msg_user_queued" not in adapter._queued_events_by_message_id
    await adapter.on_processing_complete(queued_event, types.SimpleNamespace(value="success"))
    await runner._run_agent(
        source=queued_event.source,
        session_key="vylen:chat_a:user_1",
        event_message_id="msg_user_queued",
    )

    events = adapter._chat_event_logs.get("chat_a").events
    started = [event for event in events if event.kind == "turn.started" and event.payload.get("turn_id") == "turn_normal_queue"]
    completed = [event for event in events if event.kind == "turn.completed" and event.payload.get("turn_id") == "turn_normal_queue"]
    assert len(started) == 1
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_message_steer_action_dispatches_slash_without_ack_bubble(adapter):
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        return "steer accepted"

    adapter.set_message_handler(handler)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "steer_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_steer_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.steer",
        "text": "change direction",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    user_messages = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("role") == "user"
    ]
    hermes_messages = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    ]
    assert user_messages[-1].payload["intent"] == "steer"
    assert user_messages[-1].payload["status"] == "completed"
    assert handled == ["/steer change direction"]
    assert hermes_messages == []
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert len(acks) == 1


@pytest.mark.asyncio
async def test_message_steer_action_reconciles_fallback_queue(adapter):
    session_key = "vylen:chat_a:user_1"

    async def handler(event):
        if event.text.startswith("/steer "):
            adapter._pending_messages[session_key] = FakeMessageEvent(
                text=event.text.removeprefix("/steer ").strip(),
                source=event.source,
                message_id=event.message_id,
            )
            return "Agent still starting — /steer queued for the next turn."
        return "pong"

    adapter.set_message_handler(handler)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "steer_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_steer_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.steer",
        "text": "change direction",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    updates = [
        event for event in events
        if event.kind == "message.updated" and event.payload.get("intent") == "queue"
    ]
    assert updates[-1].payload["status"] == "queued"
    assert adapter._pending_messages[session_key].text == "change direction"
    assert adapter._pending_messages[session_key].message_id == updates[-1].payload["message_id"]
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["intent"] == "queue"
    assert acks[-1]["message_status"] == "queued"

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "steer_retry",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_steer_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.steer",
        "text": "change direction",
    })

    retry_ack = [frame for frame in adapter._fake_client.sent if frame.get("request_id") == "steer_retry"][-1]
    assert retry_ack["intent"] == "queue"
    assert retry_ack["message_status"] == "queued"


@pytest.mark.asyncio
async def test_message_steer_fallback_preserves_existing_queue_head(adapter):
    session_key = "vylen:chat_a:user_1"
    source = FakeSessionSource(
        platform=adapter.platform,
        chat_id="chat_a",
        user_id="user_1",
        user_name="Giorgio",
        message_id="msg_first",
    )
    adapter._pending_messages[session_key] = FakeMessageEvent(
        text="first",
        source=source,
        message_id="msg_first",
    )

    async def handler(event):
        if event.text.startswith("/steer "):
            adapter._pending_messages[session_key] = FakeMessageEvent(
                text=event.text.removeprefix("/steer ").strip(),
                source=event.source,
                message_id=event.message_id,
            )
            return "No active agent — /steer queued for the next turn."
        return "pong"

    adapter.set_message_handler(handler)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "steer_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_steer_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.steer",
        "text": "second",
    })

    assert adapter._pending_messages[session_key].text == "first"
    assert [event.text for event in adapter._queued_events[session_key]] == ["second"]


@pytest.mark.asyncio
async def test_message_steer_success_with_same_text_queue_head_stays_steer(adapter):
    session_key = "vylen:chat_a:user_1"
    source = FakeSessionSource(
        platform=adapter.platform,
        chat_id="chat_a",
        user_id="user_1",
        user_name="Giorgio",
        message_id="msg_first",
    )
    existing = FakeMessageEvent(
        text="second",
        source=source,
        message_id="msg_first",
    )
    adapter._pending_messages[session_key] = existing

    async def handler(event):
        if event.text.startswith("/steer "):
            return "Steer queued"
        return "pong"

    adapter.set_message_handler(handler)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": "turn_active", "message_id": "msg_active"}

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "steer_req",
        "chat_id": "chat_a",
        "client_id": "phone",
        "client_message_id": "client_steer_1",
        "user_id": "user_1",
        "user_name": "Giorgio",
        "action": "message.steer",
        "text": "second",
    })

    assert adapter._pending_messages[session_key] is existing
    assert getattr(adapter, "_queued_events", {}) == {}
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["intent"] == "steer"
    assert acks[-1]["message_status"] == "completed"


@pytest.mark.asyncio
async def test_chat_message_ack_is_sent_before_long_running_handler_finishes(adapter):
    ready = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        ready.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(handler)
    task = asyncio.create_task(adapter._handle_chat_message(_chat_message_frame()))
    await ready.wait()
    await asyncio.wait_for(task, timeout=1)

    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert len(acks) == 1
    assert acks[0]["client_message_id"] == "client_msg_1"

    release.set()
    if adapter._chat_message_tasks:
        await asyncio.gather(*tuple(adapter._chat_message_tasks))


@pytest.mark.asyncio
async def test_turn_cancel_accepts_acknowledged_turn_before_processing_starts(adapter, monkeypatch):
    release = asyncio.Event()

    async def hold_processing(chat_id, user_message_id, turn_id, event):
        await release.wait()

    monkeypatch.setattr(adapter, "_process_chat_message", hold_processing)

    await adapter._handle_chat_message(_chat_message_frame())
    ack = next(frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack")
    events = adapter._chat_event_logs.get("chat_a").events
    assert not any(event.kind == "turn.started" for event in events)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_early",
        "chat_id": "chat_a",
        "turn_id": ack["turn_id"],
        "action": "turn.cancel",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors == []
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["turn_id"] == ack["turn_id"]

    release.set()
    if adapter._chat_message_tasks:
        await asyncio.gather(*tuple(adapter._chat_message_tasks))


@pytest.mark.asyncio
async def test_cancelled_turn_appends_assistant_marker_to_hermes_history(adapter):
    store = FakeSessionStore()
    adapter._session_store = store
    source = FakeSessionSource(
        platform=adapter.platform,
        chat_id="chat_a",
        user_id="user_1",
        user_name="Giorgio",
        message_id="msg_user_1",
    )
    event = FakeMessageEvent(
        text="long answer",
        source=source,
        message_id="msg_user_1",
        raw_message={"turn_id": "turn_cancelled"},
    )

    adapter._cancelled_turns.add("turn_cancelled")
    await adapter.on_processing_complete(event, types.SimpleNamespace(value="cancelled"))
    await adapter.on_processing_complete(event, types.SimpleNamespace(value="cancelled"))

    assert len(store.appended) == 1
    session_id, marker = store.appended[0]
    assert session_id == "session_chat_a_user_1"
    assert marker["role"] == "assistant"
    assert "cancelled by the user" in marker["content"]
    assert "Do not continue" in marker["content"]


@pytest.mark.asyncio
async def test_session_status_action_emits_retained_status_without_ack_bubble(adapter):
    async def initial_handler(event):
        return "pong"

    adapter.set_message_handler(initial_handler)
    await adapter._handle_chat_message(_chat_message_frame())

    async def handler(event):
        assert event.text == "/status"
        return "status text that should be suppressed"

    adapter.set_message_handler(handler)
    adapter._fake_client.sent.clear()

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "status_req",
        "chat_id": "chat_a",
        "action": "session.status",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    status_events = [event for event in events if event.kind == "session.status"]
    assert status_events
    assert status_events[-1].payload["state"] == "idle"
    assert not any(
        event.kind == "message.created" and event.payload.get("text") == "status text that should be suppressed"
        for event in events
    )
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "status_req"


@pytest.mark.asyncio
async def test_session_status_action_uses_requesting_user_source(adapter):
    async def initial_handler(event):
        return "pong"

    adapter.set_message_handler(initial_handler)
    await adapter._handle_chat_message(_chat_message_frame(user_id="user_1"))

    handled: list[Any] = []

    async def handler(event):
        handled.append(event)
        return "status text that should be suppressed"

    adapter.set_message_handler(handler)
    adapter._fake_client.sent.clear()
    adapter._pending_messages["vylen:chat_a:user_1"] = object()
    adapter._pending_messages["vylen:chat_a:user_2"] = object()

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "status_req_user_2",
        "chat_id": "chat_a",
        "action": "session.status",
        "user_id": "user_2",
        "user_name": "Ada",
    })

    assert handled[-1].source.user_id == "user_2"
    status_events = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "session.status"
    ]
    assert status_events[-1].payload["queued"] == 1


@pytest.mark.asyncio
async def test_session_status_action_returns_error_when_dispatch_fails(adapter):
    async def fail_dispatch(*args, **kwargs):
        raise RuntimeError("status failed")

    adapter._dispatch_native_command = fail_dispatch

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "status_req_failure",
        "chat_id": "chat_a",
        "action": "session.status",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "status_req_failure"
    assert errors[-1]["code"] == "SESSION_STATUS_FAILED"
    assert not any(frame["type"] == "chat_action_ack" for frame in adapter._fake_client.sent)


@pytest.mark.asyncio
async def test_session_controls_action_emits_model_and_reasoning_state(adapter):
    session_key = "vylen:chat_a:user_2"
    runner = types.SimpleNamespace(
        _session_reasoning_overrides={session_key: {"enabled": True, "effort": "high"}},
    )
    runner._resolve_session_agent_runtime = (
        lambda **kwargs: ("gpt-5.5", {"provider": "openai-codex"})
    )
    runner._resolve_session_reasoning_config = lambda **kwargs: {"enabled": True, "effort": "high"}
    adapter._runner = runner
    adapter._load_show_reasoning = lambda: True

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "controls_req",
        "chat_id": "chat_a",
        "action": "session.controls",
        "user_id": "user_2",
        "user_name": "Ada",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    controls_events = [event for event in events if event.kind == "session.controls"]
    assert controls_events
    assert controls_events[-1].payload["model"] == "gpt-5.5"
    assert controls_events[-1].payload["provider"] == "openai-codex"
    assert controls_events[-1].payload["reasoning_effort"] == "high"
    assert controls_events[-1].payload["reasoning_scope"] == "session"
    assert controls_events[-1].payload["reasoning_display"] is True
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "controls_req"


@pytest.mark.asyncio
async def test_session_reasoning_action_dispatches_hermes_reasoning_command(adapter):
    dispatched: list[str] = []
    dispatch_kwargs: list[dict[str, Any]] = []
    runner = types.SimpleNamespace(_session_reasoning_overrides={})
    runner._resolve_session_agent_runtime = (
        lambda **kwargs: ("gpt-5.5", {"provider": "openai-codex"})
    )
    runner._resolve_session_reasoning_config = lambda **kwargs: (
        {"enabled": True, "effort": "xhigh"}
        if "vylen:chat_a:user_1" in runner._session_reasoning_overrides
        else {"enabled": True, "effort": "medium"}
    )
    adapter._runner = runner

    async def dispatch(frame, chat_id, text, **kwargs):
        dispatched.append(text)
        dispatch_kwargs.append(dict(kwargs))
        runner._session_reasoning_overrides["vylen:chat_a:user_1"] = {"enabled": True, "effort": "xhigh"}
        return True

    adapter._dispatch_native_command = dispatch
    adapter._load_show_reasoning = lambda: False

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reasoning_req",
        "chat_id": "chat_a",
        "action": "session.reasoning",
        "text": "xhigh",
        "user_id": "user_1",
    })

    assert dispatched == ["/reasoning xhigh"]
    assert dispatch_kwargs[-1].get("wait_for_completion") is True
    events = adapter._chat_event_logs.get("chat_a").events
    controls_events = [event for event in events if event.kind == "session.controls"]
    assert controls_events[-1].payload["reasoning_effort"] == "xhigh"
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reasoning_req"


@pytest.mark.asyncio
async def test_session_reasoning_action_does_not_wait_for_active_session(adapter):
    dispatched: list[str] = []
    dispatch_kwargs: list[dict[str, Any]] = []
    session_key = "vylen:chat_a:user_1"
    active_task = asyncio.create_task(asyncio.sleep(60))
    adapter._session_tasks = {session_key: active_task}
    runner = types.SimpleNamespace(_session_reasoning_overrides={})
    runner._resolve_session_agent_runtime = (
        lambda **kwargs: ("gpt-5.5", {"provider": "openai-codex"})
    )
    runner._resolve_session_reasoning_config = lambda **kwargs: {"enabled": True, "effort": "medium"}
    adapter._runner = runner

    async def dispatch(frame, chat_id, text, **kwargs):
        dispatched.append(text)
        dispatch_kwargs.append(dict(kwargs))
        return True

    adapter._dispatch_native_command = dispatch
    adapter._load_show_reasoning = lambda: False

    try:
        await adapter._handle_chat_action({
            "type": "chat_action",
            "request_id": "reasoning_active_req",
            "chat_id": "chat_a",
            "action": "session.reasoning",
            "text": "high",
            "user_id": "user_1",
        })
    finally:
        active_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active_task

    assert dispatched == ["/reasoning high"]
    assert dispatch_kwargs[-1].get("wait_for_completion") is False
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reasoning_active_req"


@pytest.mark.asyncio
async def test_session_reset_action_retains_divider_and_dispatches_slash_reset(adapter):
    handled: list[str] = []

    async def initial_handler(event):
        return "pong"

    adapter.set_message_handler(initial_handler)
    await adapter._handle_chat_message(_chat_message_frame())

    async def handler(event):
        handled.append(event.text)
        return "reset text that should be suppressed"

    adapter.set_message_handler(handler)
    adapter._fake_client.sent.clear()

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req",
        "chat_id": "chat_a",
        "action": "session.reset",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    assert "/reset" in handled
    assert [event.kind for event in events].count("session.reset") == 1
    assert not any(
        event.kind == "message.created" and event.payload.get("text") == "reset text that should be suppressed"
        for event in events
    )
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reset_req"


@pytest.mark.asyncio
async def test_session_reset_action_uses_requesting_user_source(adapter):
    handled: list[Any] = []

    async def initial_handler(event):
        return "pong"

    adapter.set_message_handler(initial_handler)
    await adapter._handle_chat_message(_chat_message_frame(user_id="user_1"))

    async def handler(event):
        handled.append(event)
        return None

    adapter.set_message_handler(handler)
    adapter._fake_client.sent.clear()

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req_user_2",
        "chat_id": "chat_a",
        "action": "session.reset",
        "user_id": "user_2",
        "user_name": "Ada",
    })

    assert handled[-1].text == "/reset"
    assert handled[-1].source.user_id == "user_2"
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reset_req_user_2"


@pytest.mark.asyncio
async def test_session_reset_action_returns_error_when_dispatch_fails(adapter):
    async def fail_dispatch(*args, **kwargs):
        raise RuntimeError("reset timed out")

    adapter._dispatch_native_command = fail_dispatch

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req_failure",
        "chat_id": "chat_a",
        "action": "session.reset",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "reset_req_failure"
    assert errors[-1]["code"] == "SESSION_RESET_FAILED"
    assert adapter._chat_event_logs.get("chat_a") is None


@pytest.mark.asyncio
async def test_session_reset_action_builds_source_without_prior_message(adapter):
    handled: list[Any] = []

    async def handler(event):
        handled.append(event)
        return None

    adapter.set_message_handler(handler)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req_empty",
        "chat_id": "chat_empty",
        "action": "session.reset",
        "user_id": "user_1",
        "user_name": "Giorgio",
    })

    assert [event.text for event in handled] == ["/reset"]
    assert handled[0].source.chat_id == "chat_empty"
    assert handled[0].source.user_id == "user_1"
    assert [event.kind for event in adapter._chat_event_logs.get("chat_empty").events].count("session.reset") == 1
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reset_req_empty"


@pytest.mark.asyncio
async def test_native_reset_confirm_marker_survives_until_slash_confirm(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch)
    adapter._native_confirm_sessions["session_a"] = time.time() + adapter._action_ttl_seconds

    result = await adapter.send_slash_confirm(
        chat_id="chat_a",
        title="Confirm reset",
        message="Proceed?",
        session_key="session_a",
        confirm_id="confirm_1",
    )
    await asyncio.sleep(0)

    assert result.success
    assert calls["confirm"] == [("session_a", "confirm_1", "once", adapter._action_ttl_seconds)]
    assert "session_a" not in adapter._native_confirm_sessions
    assert adapter._chat_event_logs.get("chat_a") is None


@pytest.mark.asyncio
async def test_session_status_reports_pending_slot_queue_depth(adapter):
    async def handler(event):
        return "pong"

    adapter.set_message_handler(handler)
    await adapter._handle_chat_message(_chat_message_frame())
    adapter._fake_client.sent.clear()
    adapter._pending_messages["vylen:chat_a:user_1"] = object()

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "status_req_queue",
        "chat_id": "chat_a",
        "action": "session.status",
    })

    status_events = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "session.status"
    ]
    assert status_events[-1].payload["queued"] == 1
    assert status_events[-1].payload["queued_exact"] is False
    assert status_events[-1].payload["state"] == "queued"


@pytest.mark.asyncio
async def test_session_reset_action_suppresses_delayed_background_confirm(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch)
    confirm_started = asyncio.Event()

    async def delayed_handle_message(event):
        adapter.handled_events.append(event)

        async def later_confirm():
            await asyncio.sleep(0)
            await adapter.send_slash_confirm(
                chat_id=event.source.chat_id,
                title="Confirm reset",
                message="Proceed?",
                session_key=adapter._session_key_for_source(event.source),
                confirm_id="confirm_1",
            )
            confirm_started.set()

        session_key = adapter._session_key_for_source(event.source)
        task = asyncio.create_task(later_confirm())
        adapter._session_tasks = {session_key: task}

    adapter.handle_message = delayed_handle_message

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req_background",
        "chat_id": "chat_a",
        "action": "session.reset",
        "user_id": "user_1",
        "user_name": "Giorgio",
    })
    await asyncio.wait_for(confirm_started.wait(), timeout=1)

    assert calls["confirm"] == [("vylen:chat_a:user_1", "confirm_1", "once", adapter._action_ttl_seconds)]
    events = adapter._chat_event_logs.get("chat_a").events
    assert not any(event.kind == "confirm.requested" for event in events)


@pytest.mark.asyncio
async def test_session_reset_action_waits_for_background_command_completion(adapter, monkeypatch):
    release_resolve = asyncio.Event()
    resolve_started = asyncio.Event()
    calls = _install_fake_approval_tools(monkeypatch, confirm_wait=release_resolve, confirm_started=resolve_started)
    completed = asyncio.Event()

    async def background_command(event):
        await asyncio.sleep(0)
        await adapter.send_slash_confirm(
            chat_id=event.source.chat_id,
            title="Confirm reset",
            message="Proceed?",
            session_key=adapter._session_key_for_source(event.source),
            confirm_id="confirm_1",
        )
        completed.set()

    async def background_handle_message(event):
        adapter.handled_events.append(event)
        session_key = adapter._session_key_for_source(event.source)
        task = asyncio.create_task(background_command(event))
        adapter._session_tasks = {session_key: task}

    adapter.handle_message = background_handle_message

    reset_task = asyncio.create_task(adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "reset_req_wait",
        "chat_id": "chat_a",
        "action": "session.reset",
        "user_id": "user_1",
        "user_name": "Giorgio",
    }))
    await asyncio.wait_for(resolve_started.wait(), timeout=1)
    await asyncio.sleep(0)

    events = adapter._chat_event_logs.get("chat_a")
    assert events is None or [event.kind for event in events.events].count("session.reset") == 0
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks == []

    release_resolve.set()
    await reset_task

    assert completed.is_set()
    assert calls["confirm"] == [("vylen:chat_a:user_1", "confirm_1", "once", adapter._action_ttl_seconds)]
    events = adapter._chat_event_logs.get("chat_a").events
    assert [event.kind for event in events].count("session.reset") == 1
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "reset_req_wait"


@pytest.mark.asyncio
async def test_native_reset_confirm_marker_does_not_auto_resolve_unrelated_confirm(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch, pending_command="reload-mcp")
    adapter._native_confirm_sessions["session_a"] = {
        "deadline": time.time() + adapter._action_ttl_seconds,
        "commands": {"new"},
    }

    result = await adapter.send_slash_confirm(
        chat_id="chat_a",
        title="Reload MCP",
        message="Proceed?",
        session_key="session_a",
        confirm_id="confirm_1",
    )
    await asyncio.sleep(0)

    assert result.success
    assert calls["confirm"] == []
    assert "session_a" not in adapter._native_confirm_sessions
    events = adapter._chat_event_logs.get("chat_a").events
    assert [event.kind for event in events].count("confirm.requested") == 1


@pytest.mark.asyncio
async def test_session_status_uses_runner_queue_depth_when_available(adapter):
    async def handler(event):
        return "pong"

    adapter.set_message_handler(handler)
    await adapter._handle_chat_message(_chat_message_frame())
    adapter._fake_client.sent.clear()
    adapter._pending_messages["vylen:chat_a:user_1"] = object()
    adapter._runner = types.SimpleNamespace(_queue_depth=lambda session_key, adapter=None: 3)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "status_req_queue_depth",
        "chat_id": "chat_a",
        "action": "session.status",
    })

    status_events = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "session.status"
    ]
    assert status_events[-1].payload["queued"] == 3
    assert status_events[-1].payload["queued_exact"] is True
    assert status_events[-1].payload["state"] == "queued"


@pytest.mark.asyncio
async def test_typed_status_slash_uses_native_status_semantics(adapter):
    async def handler(event):
        assert event.text == "/status"
        return "status text that should be suppressed"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="/status"))

    events = adapter._chat_event_logs.get("chat_a").events
    assert [event.kind for event in events].count("session.status") == 1
    assert not any(
        event.kind == "message.created" and event.payload.get("text") in {"/status", "status text that should be suppressed"}
        for event in events
    )
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert acks[-1]["client_message_id"] == "client_msg_1"


@pytest.mark.asyncio
async def test_typed_status_slash_failure_removes_dedup_record(adapter):
    async def fail_dispatch(*args, **kwargs):
        raise RuntimeError("status failed")

    adapter._dispatch_native_command = fail_dispatch
    frame = _chat_message_frame(text="/status")

    await adapter._handle_chat_message(frame)

    errors = [sent for sent in adapter._fake_client.sent if sent["type"] == "chat_message_error"]
    assert errors[-1]["request_id"] == "req_1"
    assert errors[-1]["code"] == "SESSION_STATUS_FAILED"
    assert ("chat_a", "client_msg_1") not in adapter._accepted_chat_messages


@pytest.mark.asyncio
async def test_typed_reset_slash_requests_confirmation_before_reset(adapter):
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        return "reset text that should be suppressed"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="/reset"))

    events = adapter._chat_event_logs.get("chat_a").events
    assert handled == []
    assert [event.kind for event in events].count("confirm.requested") == 1
    assert [event.kind for event in events].count("session.reset") == 0
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert acks[-1]["client_message_id"] == "client_msg_1"


@pytest.mark.asyncio
async def test_typed_reset_confirm_uses_native_reset_semantics(adapter):
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        return "reset text that should be suppressed"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="/reset"))
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "confirm_reset_req",
        "chat_id": "chat_a",
        "action": "confirm.respond",
        "action_id": action_id,
        "choice": "once",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    assert handled == ["/reset"]
    assert [event.kind for event in events].count("session.reset") == 1
    assert not any(
        event.kind == "message.created" and event.payload.get("text") in {"/reset", "reset text that should be suppressed"}
        for event in events
    )
    action_acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert action_acks[-1]["request_id"] == "confirm_reset_req"


@pytest.mark.asyncio
async def test_retained_chat_events_replay_when_no_client_was_live(adapter):
    async def handler(event):
        return "pong"

    adapter.set_message_handler(handler)
    await adapter._handle_chat_message(_chat_message_frame())
    adapter._fake_client.sent.clear()

    await adapter._chat_cursors.handle_subscribe({
        "type": "chat_subscribe",
        "request_id": "sub_1",
        "chat_id": "chat_a",
        "client_id": "tablet",
        "after_seq": 0,
    })
    for _ in range(20):
        if any(frame["type"] == FRAME_CHAT_EVENT for frame in adapter._fake_client.sent):
            break
        await asyncio.sleep(0.01)

    events = [frame for frame in adapter._fake_client.sent if frame["type"] == FRAME_CHAT_EVENT]
    kinds = [frame["event"]["kind"] for frame in events]
    assert "message.created" in kinds
    assert "turn.started" in kinds
    assert "turn.completed" in kinds

    await adapter._chat_cursors.close()


@pytest.mark.asyncio
async def test_message_lifecycle_creates_and_finalizes_assistant_message(adapter):
    async def handler(event):
        return "pong"

    adapter.set_message_handler(handler)
    await adapter._handle_chat_message(_chat_message_frame())

    events = adapter._chat_event_logs.get("chat_a").events
    assistant_created = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    ]
    assert len(assistant_created) == 1

    assistant_id = assistant_created[0].payload["message_id"]
    assistant_updates = [
        event for event in events
        if event.kind == "message.updated" and event.payload.get("message_id") == assistant_id
    ]
    assert assistant_updates[-1].payload["status"] == "completed"


@pytest.mark.asyncio
async def test_processing_complete_finalizes_non_streaming_assistant_message(adapter):
    async def handler(event):
        return "pong"

    async def no_finalize_edit(*args, **kwargs):
        return FakeSendResult(success=True, message_id=str(args[1]))

    adapter.set_message_handler(handler)
    adapter.edit_message = no_finalize_edit

    await adapter._handle_chat_message(_chat_message_frame())

    events = adapter._chat_event_logs.get("chat_a").events
    assistant_id = next(
        event.payload["message_id"]
        for event in events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    )
    assistant_updates = [
        event for event in events
        if event.kind == "message.updated" and event.payload.get("message_id") == assistant_id
    ]
    assert assistant_updates[-1].payload["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_progress_is_retained_as_activity_events(adapter):
    async def handler(event):
        progress = await adapter.send(event.source.chat_id, '🔍 read_file: "Makefile"', metadata={})
        await adapter.edit_message(
            event.source.chat_id,
            progress.message_id,
            '🔍 read_file: "Makefile"\n⚙️ shell...',
        )
        return "The mobile chat files are app/mobile/app/chat/[id].tsx and shared chat state."

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="inspect files"))

    events = adapter._chat_event_logs.get("chat_a").events
    assistant_created = [
        event for event in events
        if event.kind == "message.created" and event.payload.get("role") == "hermes"
    ]
    assert len(assistant_created) == 1
    assistant_id = assistant_created[0].payload["message_id"]

    activity_started = [event for event in events if event.kind == "activity.started"]
    assert len(activity_started) >= 2
    assert activity_started[0].payload["message_id"] == assistant_id
    assert activity_started[0].payload["tool"] == "read_file"
    assert activity_started[0].payload["label"] == "Makefile"
    assert any(event.payload["tool"] == "shell" for event in activity_started)

    completed_ids = {
        event.payload["activity_id"]
        for event in events
        if event.kind == "activity.completed"
    }
    started_ids = {
        event.payload["activity_id"]
        for event in activity_started
    }
    assert started_ids <= completed_ids


@pytest.mark.asyncio
async def test_tool_progress_edits_are_coalesced_when_unchanged(adapter):
    async def handler(event):
        progress = await adapter.send(event.source.chat_id, '🔍 read_file: "Makefile"', metadata={})
        await adapter.edit_message(
            event.source.chat_id,
            progress.message_id,
            '🔍 read_file: "Makefile"',
        )
        await adapter.edit_message(
            event.source.chat_id,
            progress.message_id,
            '🔍 read_file: "docs/dev.md"',
        )
        return "done"

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="inspect files"))

    activity_events = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind.startswith("activity.")
    ]
    activity_started = [event for event in activity_events if event.kind == "activity.started"]
    activity_updated = [event for event in activity_events if event.kind == "activity.updated"]

    assert len(activity_started) == 1
    assert activity_started[0].payload["label"] == "Makefile"
    assert len(activity_updated) == 1
    assert activity_updated[0].payload["activity_id"] == activity_started[0].payload["activity_id"]
    assert activity_updated[0].payload["label"] == "docs/dev.md"


@pytest.mark.asyncio
async def test_tool_progress_marks_prior_rows_completed_before_turn_finishes(adapter):
    ready = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        progress = await adapter.send(event.source.chat_id, '👁️ vision_analyze: "Describe the image"', metadata={})
        await adapter.edit_message(
            event.source.chat_id,
            progress.message_id,
            '👁️ vision_analyze: "Describe the image"\n🎨 image_generate: "Create landscape"',
        )
        ready.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(handler)
    task = asyncio.create_task(adapter._handle_chat_message(_chat_message_frame(text="redo image")))
    await ready.wait()

    events = adapter._chat_event_logs.get("chat_a").events
    started = [event for event in events if event.kind == "activity.started"]
    completed = [event for event in events if event.kind == "activity.completed"]

    assert [event.payload["tool"] for event in started] == ["vision_analyze", "image_generate"]
    assert len(completed) == 1
    assert completed[0].payload["activity_id"] == started[0].payload["activity_id"]
    assert completed[0].payload["status"] == "completed"

    release.set()
    await task


@pytest.mark.asyncio
async def test_chat_message_decodes_inline_image_data_url(adapter, monkeypatch, tmp_path):
    seen: list[FakeMessageEvent] = []

    async def handler(event):
        seen.append(event)
        return None

    image_path = tmp_path / "whiteboard.png"
    base_mod = sys.modules["gateway.platforms.base"]
    monkeypatch.setattr(
        base_mod,
        "cache_image_from_bytes",
        lambda data, ext=".jpg": (image_path.write_bytes(data), str(image_path))[1],
    )
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(attachments=[{
        "id": "att_1",
        "type": "image",
        "mime_type": "Image/PNG",
        "filename": "whiteboard.png",
        "data_url": f"data:image/png;base64,{png}",
    }]))

    assert seen[0].message_type == FakeMessageType.PHOTO
    assert seen[0].media_urls == [str(image_path)]
    assert seen[0].media_types == ["image"]
    user_event = next(
        event
        for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "message.created" and event.payload.get("role") == "user"
    )
    attachments = user_event.payload["attachments"]
    assert attachments[0]["type"] == "image"
    assert attachments[0]["mime_type"] == "image/png"
    assert attachments[0]["filename"] == "whiteboard.png"
    assert attachments[0]["data_url"].startswith("/v1/instances/inst_1/blobs/")


@pytest.mark.asyncio
async def test_generated_image_file_is_retained_as_chat_attachment(adapter, tmp_path):
    image_path = tmp_path / "landscape.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    async def handler(event):
        await adapter.send_image_file(
            chat_id=event.source.chat_id,
            image_path=str(image_path),
            caption="Here it is:",
        )
        return None

    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(text="show me the image"))

    events = adapter._chat_event_logs.get("chat_a").events
    image_messages = [
        event for event in events
        if event.kind == "message.created"
        and event.payload.get("role") == "hermes"
        and event.payload.get("attachments")
    ]
    assert len(image_messages) == 1
    payload = image_messages[0].payload
    assert payload["text"] == "Here it is:"
    assert payload["turn_id"]
    assert payload["attachments"] == [{
        "type": "image",
        "data_url": f"/v1/instances/inst_1/blobs/{next(iter(adapter._blobs._entries))}",
        "mime_type": "image/png",
        "filename": "landscape.png",
    }]
    updates = [
        event for event in events
        if event.kind == "message.updated"
        and event.payload.get("message_id") == payload["message_id"]
    ]
    assert updates[-1].payload["status"] == "completed"


@pytest.mark.asyncio
async def test_plugin_initiated_image_push_emits_chat_index_changed(adapter, monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "to_thread", _run_sync_as_async)
    image_path = tmp_path / "landscape.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    adapter._fake_client.sent.clear()

    class FakePushRelay:
        async def send_push(self, frame):
            frame["seq"] = 1
            frame["event_id"] = "event_1"
            adapter._chat_event_logs.get_or_create(frame["chat_id"]).append("push", dict(frame))
            await adapter._fake_client.send(frame)

    adapter._chat_cursors = FakePushRelay()
    result = await adapter.send_image_file(
        chat_id="chat_a",
        image_path=str(image_path),
        caption="Here it is:",
    )

    assert result.success is True
    pushes = [frame for frame in adapter._fake_client.sent if frame["type"] == "push"]
    assert pushes
    assert pushes[-1]["chat_id"] == "inbox"
    frames = [
        frame
        for frame in adapter._fake_client.sent
        if frame["type"] == adapter_mod.FRAME_CHAT_INDEX_CHANGED
    ]
    assert frames
    assert frames[-1]["instance_id"] == "inst_1"
    assert frames[-1]["chat_id"] == "inbox"
    assert frames[-1]["chat"]["preview"]["last_message_preview"] == "Here it is:"


@pytest.mark.asyncio
async def test_turn_cancel_action_cancels_active_session(adapter):
    async def handler(event):
        if event.text == "/stop":
            return "Stopped. You can continue this session."
        return None

    adapter.set_message_handler(handler)
    await adapter._handle_chat_message(_chat_message_frame())
    turn_id = next(
        event.payload["turn_id"]
        for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "turn.started"
    )
    adapter._active_turns_by_chat["chat_a"] = {
        "turn_id": turn_id,
        "message_id": "msg_user",
        "session_key": "vylen:chat_a:user_1",
        "source": FakeSessionSource(platform=adapter.platform, chat_id="chat_a", user_id="user_1"),
    }

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_1",
        "chat_id": "chat_a",
        "turn_id": turn_id,
        "action": "turn.cancel",
    })

    handled_texts = [event.text for event in adapter.handled_events]
    assert handled_texts == ["hello", "/stop"]
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["turn_id"] == turn_id
    events = adapter._chat_event_logs.get("chat_a").events
    cancelled = [event for event in events if event.kind == "turn.cancelled"]
    assert cancelled[-1].payload["turn_id"] == turn_id
    assert cancelled[-1].payload["reason"] == "user_stop"
    placeholder = [
        event for event in events
        if event.kind == "message.created"
        and event.payload.get("role") == "hermes"
        and event.payload.get("turn_id") == turn_id
    ]
    assert placeholder[-1].payload["status"] == "cancelled"
    stop_messages = [
        event for event in events
        if event.kind.startswith("message.")
        and "Stopped. You can continue this session." in str(event.payload.get("text") or "")
    ]
    assert stop_messages == []


@pytest.mark.asyncio
async def test_turn_cancel_dispatch_failure_returns_error_and_keeps_turn_active(adapter, monkeypatch):
    turn_id = "turn_cancel_fail"
    adapter._active_turns_by_chat["chat_a"] = {
        "turn_id": turn_id,
        "message_id": "msg_user",
        "session_key": "vylen:chat_a:user_1",
        "source": FakeSessionSource(platform=adapter.platform, chat_id="chat_a", user_id="user_1"),
    }

    async def fail_dispatch(chat_id, active):
        raise RuntimeError("native stop failed")

    monkeypatch.setattr(adapter, "_dispatch_native_stop", fail_dispatch)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_fail",
        "chat_id": "chat_a",
        "turn_id": turn_id,
        "action": "turn.cancel",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["code"] == "TURN_CANCEL_FAILED"
    assert adapter._active_turns_by_chat["chat_a"]["turn_id"] == turn_id
    assert turn_id not in adapter._cancelled_turns
    assert "cancel_requested" not in adapter._active_turns_by_chat["chat_a"]
    log = adapter._chat_event_logs.get("chat_a")
    events = log.events if log is not None else []
    assert not any(event.kind == "turn.cancelled" and event.payload["turn_id"] == turn_id for event in events)


@pytest.mark.asyncio
async def test_turn_cancel_does_not_duplicate_cancel_event_when_completion_runs_during_stop(adapter, monkeypatch):
    turn_id = "turn_cancel_race"
    event = FakeMessageEvent(
        text="original",
        source=FakeSessionSource(platform=adapter.platform, chat_id="chat_a", user_id="user_1"),
        message_id="msg_user",
        raw_message={"turn_id": turn_id},
    )
    adapter._active_turns_by_chat["chat_a"] = {
        "turn_id": turn_id,
        "message_id": "msg_user",
        "session_key": "vylen:chat_a:user_1",
        "source": event.source,
    }

    async def dispatch_and_complete(chat_id, active):
        await adapter.on_processing_complete(event, types.SimpleNamespace(value="cancelled"))

    monkeypatch.setattr(adapter, "_dispatch_native_stop", dispatch_and_complete)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_race",
        "chat_id": "chat_a",
        "turn_id": turn_id,
        "action": "turn.cancel",
    })

    events = adapter._chat_event_logs.get("chat_a").events
    cancelled = [event for event in events if event.kind == "turn.cancelled"]
    assert len(cancelled) == 1
    assert cancelled[0].payload["reason"] == "user_stop"


@pytest.mark.asyncio
async def test_turn_cancel_fences_late_output_from_cancelled_task(adapter):
    ready = asyncio.Event()
    release = asyncio.Event()
    late_result: FakeSendResult | None = None

    async def handler(event):
        nonlocal late_result
        await adapter.send(event.source.chat_id, "partial answer", metadata={})
        ready.set()
        await release.wait()
        late_result = await adapter.send(event.source.chat_id, "late answer after stop", metadata={})
        return None

    adapter.set_message_handler(handler)
    task = asyncio.create_task(adapter._handle_chat_message(_chat_message_frame()))
    await ready.wait()
    turn_id = next(
        event.payload["turn_id"]
        for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "turn.started"
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_1",
        "chat_id": "chat_a",
        "turn_id": turn_id,
        "action": "turn.cancel",
    })
    release.set()
    await task
    if adapter._chat_message_tasks:
        await asyncio.gather(*tuple(adapter._chat_message_tasks))

    assert late_result is not None
    assert late_result.success is False

    events = adapter._chat_event_logs.get("chat_a").events
    assert any(event.kind == "turn.cancelled" for event in events)
    late_updates = [
        event for event in events
        if event.kind.startswith("message.") and event.payload.get("text") == "late answer after stop"
    ]
    assert late_updates == []


@pytest.mark.asyncio
async def test_new_turn_after_cancel_gets_clean_output(adapter):
    first_ready = asyncio.Event()
    first_release = asyncio.Event()
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        if event.text == "first":
            await adapter.send(event.source.chat_id, "first partial", metadata={})
            first_ready.set()
            await first_release.wait()
            await adapter.send(event.source.chat_id, "first late", metadata={})
            return None
        return "second answer"

    adapter.set_message_handler(handler)
    first_task = asyncio.create_task(adapter._handle_chat_message(_chat_message_frame(text="first")))
    await first_ready.wait()
    first_turn_id = next(
        event.payload["turn_id"]
        for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "turn.started"
    )
    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_1",
        "chat_id": "chat_a",
        "turn_id": first_turn_id,
        "action": "turn.cancel",
    })

    await adapter._handle_chat_message(_chat_message_frame(
        request_id="req_2",
        client_message_id="client_msg_2",
        text="second",
    ))
    first_release.set()
    await first_task
    if adapter._chat_message_tasks:
        await asyncio.gather(*tuple(adapter._chat_message_tasks))

    events = adapter._chat_event_logs.get("chat_a").events
    assistant_texts = [
        event.payload.get("text")
        for event in events
        if event.kind.startswith("message.") and event.payload.get("role") == "hermes"
        or (event.kind.startswith("message.") and "text" in event.payload)
    ]
    assert "first late" not in assistant_texts
    assert "second answer" in assistant_texts
    turn_ids = [
        event.payload["turn_id"]
        for event in events
        if event.kind == "turn.started"
    ]
    assert len(turn_ids) == 2
    assert turn_ids[0] != turn_ids[1]


@pytest.mark.asyncio
async def test_cancelled_processing_complete_cleans_turn_bookkeeping(adapter):
    turn_id = "turn_cancelled_cleanup"
    event = FakeMessageEvent(
        text="cancelled",
        source=FakeSessionSource(platform=adapter.platform, chat_id="chat_a", user_id="user_1"),
        message_id="msg_user",
        raw_message={"turn_id": turn_id},
    )
    adapter._cancelled_turns.add(turn_id)
    adapter._active_turns_by_chat["chat_a"] = {"turn_id": turn_id}
    adapter._assistant_messages_by_turn[turn_id] = {"msg_asst"}
    adapter._assistant_turn_by_message["msg_asst"] = turn_id
    adapter._activity_ids_by_turn[turn_id] = {"activity_1"}
    adapter._activity_payloads_by_id["activity_1"] = ("terminal", "rm", "💻")
    adapter._activity_status_by_id["activity_1"] = "running"

    await adapter.on_processing_complete(event, types.SimpleNamespace(value="success"))

    assert "chat_a" not in adapter._active_turns_by_chat
    assert turn_id not in adapter._assistant_messages_by_turn
    assert "msg_asst" not in adapter._assistant_turn_by_message
    assert turn_id not in adapter._activity_ids_by_turn
    assert "activity_1" not in adapter._activity_payloads_by_id
    assert "activity_1" not in adapter._activity_status_by_id
    assert turn_id not in adapter._cancelled_turns


@pytest.mark.asyncio
async def test_expired_approval_action_emits_retained_expiry_and_error(adapter):
    adapter._action_ttl_seconds = -1

    result = await adapter.send_exec_approval(
        chat_id="chat_a",
        command="rm -rf tmp/build",
        session_key="session_a",
        description="dangerous command",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "action_req_1",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "approval.respond",
        "choice": "once",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["code"] == "STALE_ACTION"

    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "approval.requested" in kinds
    assert "approval.expired" in kinds


@pytest.mark.asyncio
async def test_mismatched_chat_action_does_not_expire_other_chat_card(adapter):
    result = await adapter.send_exec_approval(
        chat_id="chat_a",
        command="rm -rf tmp/build",
        session_key="session_a",
        description="dangerous command",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "action_req_wrong_chat",
        "chat_id": "chat_b",
        "action_id": action_id,
        "action": "approval.respond",
        "choice": "once",
    })

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["code"] == "STALE_ACTION"
    assert action_id in adapter._action_cards

    chat_a_kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "approval.requested" in chat_a_kinds
    assert "approval.expired" not in chat_a_kinds
    assert adapter._chat_event_logs.get("chat_b") is None


@pytest.mark.asyncio
async def test_send_clarify_retains_input_requested_card(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    result = await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=["Vylen", "Hermes"],
        clarify_id="clarify_1",
        session_key="session_a",
    )

    assert result.success
    assert adapter._action_cards["clarify_1"]["kind"] == "input"
    assert adapter._action_cards["clarify_1"]["input_kind"] == "clarify"
    event = adapter._chat_event_logs.get("chat_a").events[-1]
    assert event.kind == "input.requested"
    assert event.payload["action_id"] == "clarify_1"
    assert event.payload["choices"] == ["Vylen", "Hermes"]
    assert event.payload["input_mode"] == "choice"


@pytest.mark.asyncio
async def test_clarify_input_response_resolves_gateway_callback(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=["Vylen", "Hermes"],
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.respond",
        "choice": "Hermes",
    })

    assert calls["clarify"] == [("clarify_1", "Hermes")]
    assert "clarify_1" not in adapter._action_cards
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "input.resolved" in kinds
    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["request_id"] == "clarify_req"


@pytest.mark.asyncio
async def test_choice_clarify_rejects_text_response_without_resolving(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=["Vylen", "Hermes"],
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_text_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.respond",
        "text": "Something else",
    })

    assert calls["clarify"] == [("clarify_1", "")]
    assert "clarify_1" in adapter._action_cards
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "clarify_text_req"
    assert errors[-1]["code"] == "CHAT_ACTION_INVALID"


@pytest.mark.asyncio
async def test_choice_clarify_rejects_unknown_choice_without_resolving(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=["Vylen", "Hermes"],
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_unknown_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.respond",
        "choice": "Other",
    })

    assert calls["clarify"] == [("clarify_1", "")]
    assert "clarify_1" in adapter._action_cards
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "clarify_unknown_req"
    assert errors[-1]["code"] == "INVALID_CHOICE"


@pytest.mark.asyncio
async def test_text_clarify_rejects_choice_response_without_resolving(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    await adapter.send_clarify(
        chat_id="chat_a",
        question="What token?",
        choices=None,
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_choice_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.respond",
        "choice": "token",
    })

    assert calls["clarify"] == []
    assert "clarify_1" in adapter._action_cards
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "clarify_choice_req"
    assert errors[-1]["code"] == "CHAT_ACTION_INVALID"


@pytest.mark.asyncio
async def test_clarify_input_cancel_resolves_waiter_and_cancels_turn(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    adapter._active_turns_by_chat["chat_a"] = {
        "turn_id": "turn_clarify_cancel",
        "message_id": "msg_user_cancel",
    }
    await adapter.send_clarify(
        chat_id="chat_a",
        question="What should I use?",
        choices=None,
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_cancel_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.cancel",
    })

    assert calls["clarify"] == [("clarify_1", "")]
    assert "clarify_1" not in adapter._action_cards
    events = adapter._chat_event_logs.get("chat_a").events
    kinds = [event.kind for event in events]
    assert "input.cancelled" in kinds
    assert "turn.cancelled" in kinds
    cancelled = [event for event in events if event.kind == "turn.cancelled"]
    assert cancelled[-1].payload["turn_id"] == "turn_clarify_cancel"
    assert cancelled[-1].payload["reason"] == "input_cancelled"
    messages = [event for event in events if event.kind == "message.created" and event.payload.get("role") == "hermes"]
    assert messages[-1].payload["status"] == "cancelled"
    assert "chat_a" not in adapter._active_turns_by_chat


@pytest.mark.asyncio
async def test_clarify_input_cancel_marks_retained_turn_cancelled_without_active_turn(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    adapter._active_turns_by_chat["chat_a"] = {
        "turn_id": "turn_retained_clarify_cancel",
        "message_id": "msg_user_cancel",
    }
    await adapter.send_clarify(
        chat_id="chat_a",
        question="What should I use?",
        choices=["concise", "detailed"],
        clarify_id="clarify_1",
        session_key="session_a",
    )
    adapter._active_turns_by_chat.pop("chat_a", None)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_cancel_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.cancel",
    })

    assert calls["clarify"] == [("clarify_1", "")]
    events = adapter._chat_event_logs.get("chat_a").events
    cancelled = [event for event in events if event.kind == "turn.cancelled"]
    assert cancelled[-1].payload["turn_id"] == "turn_retained_clarify_cancel"
    assert cancelled[-1].payload["reason"] == "input_cancelled"


@pytest.mark.asyncio
async def test_expired_clarify_input_emits_retained_expiry_and_error(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls = _install_fake_clarify_tools(monkeypatch)
    adapter._action_ttl_seconds = -1
    await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=["Vylen", "Hermes"],
        clarify_id="clarify_1",
        session_key="session_a",
    )

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "clarify_stale_req",
        "chat_id": "chat_a",
        "action_id": "clarify_1",
        "action": "input.respond",
        "choice": "Vylen",
    })

    assert calls["clarify"] == [("clarify_1", "")]
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["code"] == "STALE_ACTION"
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "input.requested" in kinds
    assert "input.expired" in kinds


@pytest.mark.asyncio
async def test_update_prompt_response_writes_hermes_update_response(adapter, monkeypatch, tmp_path):
    _use_immediate_to_thread(monkeypatch)
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    (tmp_path / ".update_prompt.json").write_text("{}")
    runner = types.SimpleNamespace(_update_prompt_pending={"session_a": True})

    async def runner_handler(self, event):
        return None

    runner._handle_message = runner_handler.__get__(runner, type(runner))
    adapter.set_message_handler(runner._handle_message)

    result = await adapter.send_update_prompt(
        chat_id="chat_a",
        prompt="Apply config migration?",
        default="n",
        session_key="session_a",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "update_req",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "input.respond",
        "choice": "y",
    })

    assert (tmp_path / ".update_response").read_text() == "y"
    assert not (tmp_path / ".update_prompt.json").exists()
    assert runner._update_prompt_pending == {}
    assert action_id not in adapter._action_cards
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "input.resolved" in kinds


@pytest.mark.asyncio
async def test_update_prompt_cancel_writes_safe_denial(adapter, monkeypatch, tmp_path):
    _use_immediate_to_thread(monkeypatch)
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    (tmp_path / ".update_prompt.json").write_text("{}")
    runner = types.SimpleNamespace(_update_prompt_pending={"session_a": True})

    async def runner_handler(self, event):
        return None

    runner._handle_message = runner_handler.__get__(runner, type(runner))
    adapter.set_message_handler(runner._handle_message)

    result = await adapter.send_update_prompt(
        chat_id="chat_a",
        prompt="Apply config migration?",
        default="y",
        session_key="session_a",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "update_cancel_req",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "input.cancel",
    })

    assert (tmp_path / ".update_response").read_text() == "n"
    assert not (tmp_path / ".update_prompt.json").exists()
    assert runner._update_prompt_pending == {}
    assert action_id not in adapter._action_cards
    events = adapter._chat_event_logs.get("chat_a").events
    cancelled = [event for event in events if event.kind == "input.cancelled"]
    assert cancelled[-1].payload["response_text"] == "n"


@pytest.mark.asyncio
async def test_expired_update_prompt_writes_safe_denial_and_clears_pending(adapter, monkeypatch, tmp_path):
    _use_immediate_to_thread(monkeypatch)
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    (tmp_path / ".update_prompt.json").write_text("{}")
    runner = types.SimpleNamespace(_update_prompt_pending={"session_a": True})

    async def runner_handler(self, event):
        return None

    runner._handle_message = runner_handler.__get__(runner, type(runner))
    adapter.set_message_handler(runner._handle_message)
    adapter._action_ttl_seconds = -1

    result = await adapter.send_update_prompt(
        chat_id="chat_a",
        prompt="Apply config migration?",
        default="y",
        session_key="session_a",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "update_expired_req",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "input.respond",
        "choice": "yes",
    })

    assert (tmp_path / ".update_response").read_text() == "n"
    assert not (tmp_path / ".update_prompt.json").exists()
    assert runner._update_prompt_pending == {}
    assert action_id not in adapter._action_cards
    events = adapter._chat_event_logs.get("chat_a").events
    expired = [event for event in events if event.kind == "input.expired"]
    assert expired[-1].payload["response_text"] == "n"
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["request_id"] == "update_expired_req"
    assert errors[-1]["code"] == "STALE_ACTION"


@pytest.mark.asyncio
async def test_update_prompt_pending_does_not_capture_composer_message(adapter, monkeypatch, tmp_path):
    _use_immediate_to_thread(monkeypatch)
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    source = adapter._source_for_chat_action(_chat_message_frame(), "chat_a")
    session_key = adapter._session_key_for_source(source)
    handled: list[str] = []

    class FakeRunner:
        def __init__(self) -> None:
            self._update_prompt_pending = {session_key: True}

        async def _handle_message(self, event):
            key = adapter._session_key_for_source(event.source)
            if self._update_prompt_pending.get(key):
                (tmp_path / ".update_response").write_text(event.text)
                self._update_prompt_pending.pop(key, None)
                return
            handled.append(event.text)

    runner = FakeRunner()

    result = await adapter.send_update_prompt(
        chat_id="chat_a",
        prompt="Apply config migration?",
        default="n",
        session_key=session_key,
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    adapter.set_message_handler(runner._handle_message)
    release_runner = asyncio.Event()
    runner_tasks: list[asyncio.Task] = []

    async def start_background_runner(event):
        async def run_later():
            await release_runner.wait()
            await adapter._message_handler(event)

        runner_tasks.append(asyncio.create_task(run_later()))

    adapter.handle_message = start_background_runner

    await adapter._handle_chat_message(_chat_message_frame(text="yes"))
    release_runner.set()
    await asyncio.wait_for(runner_tasks[0], timeout=1)

    assert handled == ["yes"]
    assert not (tmp_path / ".update_response").exists()
    assert runner._update_prompt_pending == {session_key: True}
    assert action_id in adapter._action_cards


@pytest.mark.asyncio
async def test_clarify_pending_does_not_capture_composer_message(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    tools_mod = types.ModuleType("tools")
    clarify_gateway_mod = types.ModuleType("tools.clarify_gateway")
    source = adapter._source_for_chat_action(_chat_message_frame(), "chat_a")
    session_key = adapter._session_key_for_source(source)
    entry = types.SimpleNamespace(
        clarify_id="clarify_1",
        session_key=session_key,
        awaiting_text=True,
        event=asyncio.Event(),
    )
    captured: list[str] = []
    handled: list[str] = []

    def get_pending_for_session(key):
        if key == session_key and entry.awaiting_text:
            return entry
        return None

    clarify_gateway_mod.get_pending_for_session = get_pending_for_session
    tools_mod.clarify_gateway = clarify_gateway_mod
    monkeypatch.setitem(sys.modules, "tools", tools_mod)
    monkeypatch.setitem(sys.modules, "tools.clarify_gateway", clarify_gateway_mod)

    result = await adapter.send_clarify(
        chat_id="chat_a",
        question="Which workspace?",
        choices=None,
        clarify_id="clarify_1",
        session_key=session_key,
    )
    assert result.success

    async def runner_handler(event):
        pending = clarify_gateway_mod.get_pending_for_session(adapter._session_key_for_source(event.source))
        if pending is not None:
            captured.append(event.text)
            pending.event.set()
            return
        handled.append(event.text)

    adapter.set_message_handler(runner_handler)
    release_runner = asyncio.Event()
    runner_tasks: list[asyncio.Task] = []

    async def start_background_runner(event):
        async def run_later():
            await release_runner.wait()
            await adapter._message_handler(event)

        runner_tasks.append(asyncio.create_task(run_later()))

    adapter.handle_message = start_background_runner

    await adapter._handle_chat_message(_chat_message_frame(text="Vylen"))
    release_runner.set()
    await asyncio.wait_for(runner_tasks[0], timeout=1)

    assert captured == []
    assert handled == ["Vylen"]
    assert entry.awaiting_text is True
    assert "clarify_1" in adapter._action_cards


@pytest.mark.asyncio
async def test_model_picker_requires_explicit_provider_then_model_selection(adapter, monkeypatch):
    _use_immediate_to_thread(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    async def on_model_selected(chat_id, model_id, provider_slug):
        calls.append((chat_id, model_id, provider_slug))
        return f"Switched to {model_id}"

    result = await adapter.send_model_picker(
        chat_id="chat_a",
        providers=[{
            "slug": "openai",
            "name": "OpenAI",
            "models": ["gpt-5", "gpt-5.1"],
            "total_models": 2,
        }],
        current_model="gpt-5",
        current_provider="openai",
        session_key="session_a",
        on_model_selected=on_model_selected,
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "model_provider_req",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "input.respond",
        "choice": "provider:openai",
    })

    assert calls == []
    assert adapter._action_cards[action_id]["stage"] == "model"
    model_events = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "input.requested" and event.payload.get("stage") == "model"
    ]
    assert model_events

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "model_select_req",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "input.respond",
        "choice": "model:1",
    })

    assert calls == [("chat_a", "gpt-5.1", "openai")]
    assert action_id not in adapter._action_cards
    resolved = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "input.resolved"
    ]
    assert resolved[-1].payload["selected_model"] == "gpt-5.1"


def _use_immediate_to_thread(monkeypatch):
    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(adapter_mod.asyncio, "to_thread", immediate_to_thread)


def _install_fake_approval_tools(
    monkeypatch,
    *,
    approval_count: int = 1,
    confirm_follow_up: str | None = None,
    pending_command: str = "new",
    confirm_wait: asyncio.Event | None = None,
    confirm_started: asyncio.Event | None = None,
):
    tools_mod = types.ModuleType("tools")
    approval_mod = types.ModuleType("tools.approval")
    slash_confirm_mod = types.ModuleType("tools.slash_confirm")
    calls: dict[str, list[tuple[Any, ...]]] = {"approval": [], "confirm": []}
    pending_confirms: dict[str, dict[str, Any]] = {
        "session_a": {"confirm_id": "confirm_1", "command": pending_command},
        "vylen:chat_a:user_1": {"confirm_id": "confirm_1", "command": pending_command},
    }

    def resolve_gateway_approval(session_key, choice):
        calls["approval"].append((session_key, choice))
        return approval_count

    def get_pending(session_key):
        pending = pending_confirms.get(session_key)
        return dict(pending) if pending else None

    async def resolve(session_key, confirm_id, choice, timeout=300):
        if confirm_started is not None:
            confirm_started.set()
        if confirm_wait is not None:
            await confirm_wait.wait()
        calls["confirm"].append((session_key, confirm_id, choice, timeout))
        pending_confirms.pop(session_key, None)
        return confirm_follow_up

    approval_mod.resolve_gateway_approval = resolve_gateway_approval
    slash_confirm_mod.get_pending = get_pending
    slash_confirm_mod.resolve = resolve
    tools_mod.approval = approval_mod
    tools_mod.slash_confirm = slash_confirm_mod
    monkeypatch.setitem(sys.modules, "tools", tools_mod)
    monkeypatch.setitem(sys.modules, "tools.approval", approval_mod)
    monkeypatch.setitem(sys.modules, "tools.slash_confirm", slash_confirm_mod)
    return calls


def _install_fake_clarify_tools(monkeypatch, *, resolve: bool = True):
    tools_mod = types.ModuleType("tools")
    clarify_gateway_mod = types.ModuleType("tools.clarify_gateway")
    calls: dict[str, list[tuple[Any, ...]]] = {"clarify": []}

    def resolve_gateway_clarify(clarify_id, response):
        calls["clarify"].append((clarify_id, response))
        return resolve

    clarify_gateway_mod.resolve_gateway_clarify = resolve_gateway_clarify
    tools_mod.clarify_gateway = clarify_gateway_mod
    monkeypatch.setitem(sys.modules, "tools", tools_mod)
    monkeypatch.setitem(sys.modules, "tools.clarify_gateway", clarify_gateway_mod)
    return calls


def _install_hermes_session_sentinels(adapter):
    adapter._running_agents = {"session_a": "running-agent-sentinel"}
    adapter._active_sessions = {"session_a": "active-session-sentinel"}
    adapter._session_run_generation = {"session_a": 7}
    adapter._pending_messages = {"session_a": "pending-message-sentinel"}
    return {
        "running_agents": dict(adapter._running_agents),
        "active_sessions": dict(adapter._active_sessions),
        "session_run_generation": dict(adapter._session_run_generation),
        "pending_messages": dict(adapter._pending_messages),
    }


def _assert_hermes_session_sentinels_unchanged(adapter, before):
    assert adapter._running_agents == before["running_agents"]
    assert adapter._active_sessions == before["active_sessions"]
    assert adapter._session_run_generation == before["session_run_generation"]
    assert adapter._pending_messages == before["pending_messages"]


@pytest.mark.asyncio
async def test_approval_response_resolves_hermes_callback_without_session_mutation(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch)
    result = await adapter.send_exec_approval(
        chat_id="chat_a",
        command="rm -rf tmp/build",
        session_key="session_a",
        description="dangerous command",
    )
    assert result.success
    action_id = next(iter(adapter._action_cards))
    before = _install_hermes_session_sentinels(adapter)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "action_req_approval",
        "chat_id": "chat_a",
        "action_id": action_id,
        "action": "approval.respond",
        "choice": "once",
    })

    assert calls["approval"] == [("session_a", "once")]
    _assert_hermes_session_sentinels_unchanged(adapter, before)
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "approval.resolved" in kinds


@pytest.mark.asyncio
async def test_confirm_response_resolves_hermes_callback_without_session_mutation(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch)
    result = await adapter.send_slash_confirm(
        chat_id="chat_a",
        title="Confirm action",
        message="Proceed?",
        session_key="session_a",
        confirm_id="confirm_1",
    )
    assert result.success
    before = _install_hermes_session_sentinels(adapter)

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "action_req_confirm",
        "chat_id": "chat_a",
        "action_id": "confirm_1",
        "action": "confirm.respond",
        "choice": "once",
    })

    assert calls["confirm"] == [("session_a", "confirm_1", "once", adapter._action_ttl_seconds)]
    _assert_hermes_session_sentinels_unchanged(adapter, before)
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "confirm.resolved" in kinds
    follow_ups = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind.startswith("message.") and event.payload.get("text") == "Confirmed."
    ]
    assert follow_ups == []


@pytest.mark.asyncio
async def test_confirm_response_errors_when_hermes_pending_callback_is_missing(adapter, monkeypatch):
    calls = _install_fake_approval_tools(monkeypatch)
    result = await adapter.send_slash_confirm(
        chat_id="chat_a",
        title="Confirm action",
        message="Proceed?",
        session_key="missing_session",
        confirm_id="confirm_1",
    )
    assert result.success

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "action_req_confirm_stale",
        "chat_id": "chat_a",
        "action_id": "confirm_1",
        "action": "confirm.respond",
        "choice": "once",
    })

    assert calls["confirm"] == []
    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_error"]
    assert errors[-1]["code"] == "STALE_ACTION"
    kinds = [event.kind for event in adapter._chat_event_logs.get("chat_a").events]
    assert "confirm.expired" in kinds


@pytest.mark.asyncio
async def test_chat_message_reports_chat_state_unavailable(adapter):
    adapter._chat_event_logs._unavailable = ChatStateUnavailable(
        "chat_state_unavailable",
        "schema version is newer",
    )

    await adapter._handle_chat_message(_chat_message_frame())

    errors = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_error"]
    assert errors[-1]["code"] == "chat_state_unavailable"


@pytest.mark.asyncio
async def test_chat_delete_action_appends_tombstone(adapter, monkeypatch):
    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(adapter_mod.asyncio, "to_thread", immediate_to_thread)
    await adapter._append_chat_event_async("chat_a", "message.created", {"text": "hello"})

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "delete_req",
        "chat_id": "chat_a",
        "action": "chat.delete",
    })

    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["request_id"] == "delete_req"
    chat = adapter._chat_event_logs.get_chat("chat_a", include_deleted=True)
    assert chat.deleted_at is not None
    assert adapter._chat_event_logs.replay_after("chat_a", 0)[-1].kind == "chat.deleted"


@pytest.mark.asyncio
async def test_chat_rename_action_persists_event_and_syncs_hermes_title(adapter, monkeypatch):
    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(adapter_mod.asyncio, "to_thread", immediate_to_thread)
    await adapter._append_chat_event_async("chat_a", "message.created", {"text": "hello", "role": "user"})
    db = FakeSessionDB()
    store = FakeSessionStore()
    store._db = db
    adapter._session_store = store

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "rename_req",
        "chat_id": "chat_a",
        "action": "chat.rename",
        "title": "  Launch   plan  ",
        "user_id": "user_1",
        "user_name": "Giorgio",
    })

    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_action_ack"]
    assert acks[-1]["request_id"] == "rename_req"
    assert acks[-1]["title"] == "Launch   plan"
    assert adapter._chat_event_logs.get_chat("chat_a").title == "Launch   plan"
    assert adapter._chat_event_logs.replay_after("chat_a", 0)[-1].kind == "chat.renamed"
    assert db.titles == [("session_chat_a_user_1", "Launch plan")]


@pytest.mark.asyncio
async def test_native_title_message_persists_rename_without_user_echo(adapter, monkeypatch):
    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(adapter_mod.asyncio, "to_thread", immediate_to_thread)

    await adapter._handle_chat_message(_chat_message_frame(
        request_id="title_req",
        client_message_id="client_title_1",
        text="/title Mobile plan",
    ))

    acks = [frame for frame in adapter._fake_client.sent if frame["type"] == "chat_message_ack"]
    assert acks[-1]["request_id"] == "title_req"
    assert adapter._chat_event_logs.get_chat("chat_a").title == "Mobile plan"
    events = adapter._chat_event_logs.replay_after("chat_a", 0)
    assert [event.kind for event in events] == ["chat.renamed"]


@pytest.mark.asyncio
async def test_append_threshold_sweep_schedules_from_event_loop(adapter):
    class SweepRegistry:
        def __init__(self) -> None:
            self.consumed = 0
            self.swept = 0

        def consume_sweep_requested(self) -> bool:
            self.consumed += 1
            return True

        def sweep(self) -> int:
            self.swept += 1
            return 0

    registry = SweepRegistry()
    adapter._chat_event_logs = registry

    adapter._maybe_schedule_chat_state_sweep()
    await adapter._chat_sweep_request_task

    assert registry.consumed == 1
    assert registry.swept == 1
