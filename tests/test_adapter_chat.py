from __future__ import annotations

import base64
import asyncio
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

import hermes_vylen_gateway.adapter as adapter_mod
from hermes_vylen_gateway.chat_cursor import FRAME_CHAT_EVENT, ChatCursorRelay


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(dict(frame))


class FakePlatform:
    def __init__(self, value: str) -> None:
        self.value = value


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
        self.cancelled_sessions: list[str] = []

    def set_message_handler(self, handler) -> None:
        self._message_handler = handler

    async def handle_message(self, event) -> None:
        if self._message_handler is None:
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


@pytest.fixture
def adapter(monkeypatch):
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

    user_echoes = [
        event for event in adapter._chat_event_logs.get("chat_a").events
        if event.kind == "message.created" and event.payload["role"] == "user"
    ]
    assert len(user_echoes) == 1
    assert user_echoes[0].payload["client_message_id"] == "client_msg_1"


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
    await asyncio.sleep(0)

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
async def test_chat_message_decodes_inline_image_data_url(adapter):
    seen: list[FakeMessageEvent] = []

    async def handler(event):
        seen.append(event)
        return None

    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
    adapter.set_message_handler(handler)

    await adapter._handle_chat_message(_chat_message_frame(attachments=[{
        "id": "att_1",
        "type": "image",
        "mime_type": "image/png",
        "filename": "whiteboard.png",
        "data_url": f"data:image/png;base64,{png}",
    }]))

    assert seen[0].message_type == FakeMessageType.PHOTO
    assert seen[0].media_urls == ["/tmp/fake-image.png"]
    assert seen[0].media_types == ["image"]


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
async def test_turn_cancel_action_cancels_active_session(adapter):
    async def handler(event):
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
    }

    await adapter._handle_chat_action({
        "type": "chat_action",
        "request_id": "cancel_req_1",
        "chat_id": "chat_a",
        "turn_id": turn_id,
        "action": "turn.cancel",
    })

    assert adapter.cancelled_sessions == ["vylen:chat_a:user_1"]
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
