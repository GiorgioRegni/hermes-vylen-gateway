"""VylenGatewayAdapter — a Hermes BasePlatformAdapter that proxies between
the agent and Vylen Cloud over the gateway WebSocket.

This module imports from `hermes_agent.*` lazily inside the class body so the
package itself remains importable when Hermes is not installed (the doctor CLI
relies on this).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional

from .blobs import BlobRegistry
from .chat_cursor import (
    FRAME_CHAT_LIST,
    FRAME_CHAT_SNAPSHOT,
    FRAME_CHAT_SUBSCRIBE,
    FRAME_CHAT_UNSUBSCRIBE,
    ChatCursorRelay,
)
from .chat_store import ChatStateStore, ChatStateUnavailable, InvalidChatStateEvent
from .client import HandshakeError, VylenGatewayClient
from .config import ConfigError, load_from_env
from .event_log import EventTooLarge
from .health import HealthReporter
from .memory import FRAME_MEMORY_REQUEST, MemoryRPC
from .relay import FRAME_REQUEST, FRAME_RESPONSE_RESUME, HermesRelay
from .response_buffer import ResponseBufferRegistry
from .transcribe import FRAME_TRANSCRIBE, Transcriber

logger = logging.getLogger(__name__)

VYLEN_INBOX_CHAT_ID = "inbox"
FRAME_CHAT_MESSAGE = "chat_message"
FRAME_CHAT_MESSAGE_ACK = "chat_message_ack"
FRAME_CHAT_MESSAGE_ERROR = "chat_message_error"
FRAME_CHAT_ACTION = "chat_action"
FRAME_CHAT_ACTION_ACK = "chat_action_ack"
FRAME_CHAT_ACTION_ERROR = "chat_action_error"
VYLEN_ALLOWED_USERS_ENV = "VYLEN_ALLOWED_USERS"

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_TOOL_PROGRESS_LINE_RE = re.compile(
    r"^(?P<emoji>\S+)\s+(?P<tool>[A-Za-z_][A-Za-z0-9_.-]*)(?:(?P<ellipsis>\.\.\.)|:\s*(?P<label>.*))$"
)
_DEDUP_SUFFIX_RE = re.compile(r"\s+\(×\d+\)$")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s", name, raw, default)
        return default


async def _sweep_loop(
    registry: Any, interval_seconds: float, *, label: str = "registry"
) -> None:
    """Periodically drop completed/expired in-memory gateway state."""
    if interval_seconds <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise
        try:
            if hasattr(registry, "db_live_bytes"):
                evicted = await asyncio.to_thread(registry.sweep)
            else:
                evicted = registry.sweep()
            if evicted:
                logger.debug("%s sweep: evicted=%d", label, evicted)
        except Exception:  # noqa: BLE001
            logger.exception("%s sweep failed", label)


async def _run_single_sweep(registry: Any, *, label: str = "registry") -> None:
    try:
        if hasattr(registry, "db_live_bytes"):
            evicted = await asyncio.to_thread(registry.sweep)
        else:
            evicted = registry.sweep()
        if evicted:
            logger.debug("%s sweep: evicted=%d", label, evicted)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("%s sweep failed", label)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %s", name, raw, default)
        return default


def _derive_chat_title_from_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:60] if text else ""

# Cron output reaches `BasePlatformAdapter.send()` wrapped in this envelope
# (built by `cron/scheduler.py` around line 503; controlled by the
# `cron.wrap_response` config key, on by default). We strip it before
# forwarding to the Vylen UI and extract the structured `job_id`/`name`
# so the reply path can label and attribute pushes.
#
# Shape (job_id + name are required; the "To stop" footer is best-effort):
#   Cronjob Response: <name>
#   (job_id: <id>)
#   -------------
#
#   <actual content>
#
#   To stop or manage this job, send me a new message ...
#
# Tracked as backlog: ask upstream Hermes to pass these as `metadata`
# instead of regex-parsing. See vylen/docs/invariants.md.
_CRON_ENVELOPE_RE = re.compile(
    r"\ACronjob Response:\s*(?P<name>.*?)\n"
    r"\(job_id:\s*(?P<job_id>[^\)]+)\)\n"
    r"-+\n+"
    r"(?P<body>.*?)"
    r"(?:\n+To stop or manage this job[^\n]*)?\Z",
    re.DOTALL,
)


def _import_hermes():
    """Lazy import the Hermes pieces we extend. Raises ImportError if missing."""
    from gateway.platforms.base import BasePlatformAdapter, Platform  # noqa: F401
    return BasePlatformAdapter, Platform


def make_adapter_class():
    """Build the adapter class lazily so module import doesn't require Hermes."""
    BasePlatformAdapter, Platform = _import_hermes()

    class VylenGatewayAdapter(BasePlatformAdapter):
        """Hermes side of the Vylen Cloud gateway WebSocket.

        Checkpoint 3 implements the connect / disconnect / handshake path
        only. send() is a stub that will be filled in at checkpoint 4 when
        message routing lands.
        """

        REQUIRES_EDIT_FINALIZE = True

        def __init__(self, config, platform=None):
            # Platform("vylen") goes through Platform._missing_ which returns
            # the pseudo-member the platform_registry already created when
            # register(ctx) ran. Identity-stable across calls.
            super().__init__(config, platform or Platform("vylen"))
            self._client: VylenGatewayClient | None = None
            self._task: asyncio.Task | None = None
            self._instance_id: str | None = None
            self._relay: HermesRelay | None = None
            self._health: HealthReporter | None = None
            self._transcribe: Transcriber | None = None
            self._memory: MemoryRPC | None = None
            self._chat_cursors: ChatCursorRelay | None = None
            self._chat_event_logs = ChatStateStore.from_env()
            self._accepted_chat_messages: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
            self._accepted_chat_messages_max = _env_int("VYLEN_CHAT_MESSAGE_DEDUP_MAX", 5000)
            self._active_turns_by_chat: dict[str, dict[str, Any]] = {}
            self._turns_by_task: dict[asyncio.Task, dict[str, Any]] = {}
            self._cancelled_turns: set[str] = set()
            self._session_history_cancel_markers: set[str] = set()
            self._suppress_response_tasks: set[asyncio.Task] = set()
            self._native_confirm_sessions: dict[str, dict[str, Any]] = {}
            self._chat_message_tasks: set[asyncio.Task] = set()
            self._assistant_turn_by_message: dict[str, str] = {}
            self._assistant_messages_by_turn: dict[str, set[str]] = {}
            self._activity_groups_by_message: dict[str, list[dict[str, str]]] = {}
            self._activity_ids_by_turn: dict[str, set[str]] = {}
            self._activity_payloads_by_id: dict[str, tuple[str, str, str]] = {}
            self._activity_status_by_id: dict[str, str] = {}
            self._activity_message_by_turn: dict[str, str] = {}
            self._action_cards: OrderedDict[str, dict[str, Any]] = OrderedDict()
            self._action_ttl_seconds = _env_float("VYLEN_CHAT_ACTION_TTL_SECONDS", 300.0)
            self._attachment_max_bytes = _env_int("VYLEN_CHAT_ATTACHMENT_MAX_BYTES", 5 * 1024 * 1024)
            self._attachment_total_max_bytes = _env_int("VYLEN_CHAT_ATTACHMENT_TOTAL_MAX_BYTES", 10 * 1024 * 1024)
            # One blob registry per adapter lifetime; entries auto-expire
            # (see blobs.py). Chat cursor logs also live for the adapter
            # lifetime, so replayed image pushes must keep resolving tokens
            # across gateway socket reconnects.
            self._blobs = BlobRegistry()
            self._response_buffers: ResponseBufferRegistry | None = None
            self._response_sweep_task: asyncio.Task | None = None
            self._chat_sweep_task: asyncio.Task | None = None
            self._chat_sweep_request_task: asyncio.Task | None = None
            self._stopping = False

        async def connect(self) -> bool:
            try:
                load_from_env()
            except ConfigError as exc:
                logger.error("Vylen gateway config invalid: %s", exc)
                return False
            self._stopping = False
            # Start the supervisor; it owns the WS lifecycle and reconnects
            # the socket on every drop. Initial dial happens in the loop so
            # connect() returns True immediately even if the cloud is briefly
            # unreachable at boot.
            self._task = asyncio.create_task(self._supervisor())
            return True

        async def disconnect(self) -> None:
            self._stopping = True
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._task = None
            await self._teardown_session()
            self._chat_event_logs.close()

        async def _supervisor(self) -> None:
            backoff = 1.0
            while not self._stopping:
                if not await self._open_session():
                    # Failed to dial. Backoff up to 60s.
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 60.0)
                    continue
                backoff = 1.0
                # Pump frames until the socket dies, then loop and reconnect.
                assert self._client is not None and self._relay is not None
                relay = self._relay
                transcriber = self._transcribe
                memory = self._memory

                async def on_frame(frame):
                    t = frame.get("type")
                    if t == FRAME_REQUEST:
                        await relay.handle(frame)
                    elif t == FRAME_RESPONSE_RESUME:
                        await relay.handle_resume(frame)
                    elif t == FRAME_CHAT_SUBSCRIBE and self._chat_cursors is not None:
                        await self._chat_cursors.handle_subscribe(frame)
                    elif t == FRAME_CHAT_UNSUBSCRIBE and self._chat_cursors is not None:
                        self._chat_cursors.cancel(str(frame.get("request_id") or ""))
                    elif t == FRAME_CHAT_LIST and self._chat_cursors is not None:
                        await self._chat_cursors.handle_list(frame)
                    elif t == FRAME_CHAT_SNAPSHOT and self._chat_cursors is not None:
                        await self._chat_cursors.handle_snapshot(frame)
                    elif t == FRAME_CHAT_MESSAGE:
                        await self._handle_chat_message(frame)
                    elif t == FRAME_CHAT_ACTION:
                        await self._handle_chat_action(frame)
                    elif t == FRAME_TRANSCRIBE and transcriber is not None:
                        await transcriber.handle(frame)
                    elif t == FRAME_MEMORY_REQUEST and memory is not None:
                        await memory.handle(frame)

                try:
                    await self._client.iter_frames(on_frame)
                except Exception as exc:  # noqa: BLE001
                    logger.info("Vylen gateway socket dropped: %s", exc)
                await self._teardown_session()
                if self._stopping:
                    return
                logger.info("Vylen gateway reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)

        async def _open_session(self) -> bool:
            try:
                gateway_cfg = load_from_env()
            except ConfigError as exc:
                logger.error("Vylen gateway config invalid: %s", exc)
                return False
            client = VylenGatewayClient(gateway_cfg)
            try:
                ready = await client.connect()
            except HandshakeError as exc:
                logger.warning("Vylen gateway handshake failed: %s", exc)
                await client.close()
                return False
            self._client = client
            self._instance_id = ready.instance_id
            if hasattr(self._chat_event_logs, "set_event_loop"):
                self._chat_event_logs.set_event_loop(asyncio.get_running_loop())
            _authorize_vylen_user(ready.user_id)
            self._response_buffers = ResponseBufferRegistry(
                grace_seconds=_env_float("VYLEN_RESUME_GRACE_SECONDS", 300.0),
                max_bytes=_env_int("VYLEN_RESUME_MAX_BYTES", 4 * 1024 * 1024),
            )
            sweep_interval = _env_float("VYLEN_RESUME_SWEEP_SECONDS", 60.0)
            self._response_sweep_task = asyncio.create_task(
                _sweep_loop(self._response_buffers, sweep_interval, label="response buffer")
            )
            chat_sweep_interval = _env_float("VYLEN_CHAT_STATE_GC_INTERVAL_SECONDS", 3600.0)
            self._chat_sweep_task = asyncio.create_task(
                _sweep_loop(self._chat_event_logs, chat_sweep_interval, label="chat event log")
            )
            self._relay = HermesRelay(
                client.send,
                blobs=self._blobs,
                response_buffers=self._response_buffers,
            )
            self._chat_cursors = ChatCursorRelay(
                client.send,
                self._chat_event_logs,
                disabled=bool(os.environ.get("VYLEN_CHAT_CURSOR_DISABLE")),
            )
            self._health = HealthReporter(client.send, chat_state_status=lambda: self._chat_event_logs.status)
            self._health.start()
            self._transcribe = Transcriber(client.send)
            self._memory = MemoryRPC(client.send)
            logger.info(
                "Vylen gateway online: instance_id=%s user_id=%s hermes=in-process",
                ready.instance_id, ready.user_id,
            )
            return True

        async def _teardown_session(self) -> None:
            if self._response_sweep_task:
                self._response_sweep_task.cancel()
                try:
                    await self._response_sweep_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._response_sweep_task = None
            if self._chat_sweep_task:
                self._chat_sweep_task.cancel()
                try:
                    await self._chat_sweep_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._chat_sweep_task = None
            if self._chat_sweep_request_task:
                self._chat_sweep_request_task.cancel()
                try:
                    await self._chat_sweep_request_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._chat_sweep_request_task = None
            for task in tuple(self._chat_message_tasks):
                task.cancel()
            if self._chat_message_tasks:
                await asyncio.gather(*self._chat_message_tasks, return_exceptions=True)
                self._chat_message_tasks.clear()
            if self._memory:
                await self._memory.close()
                self._memory = None
            if self._transcribe:
                await self._transcribe.close()
                self._transcribe = None
            if self._health:
                await self._health.stop()
                self._health = None
            if self._relay:
                await self._relay.close()
                self._relay = None
            if self._chat_cursors:
                await self._chat_cursors.close()
                self._chat_cursors = None
            self._response_buffers = None
            if self._client:
                await self._client.close()
                self._client = None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            from gateway.platforms.base import SendResult

            current_task = asyncio.current_task()
            if (
                (current_task is not None and current_task in self._suppress_response_tasks)
                or str(reply_to or "").startswith("msg_command_")
            ):
                return SendResult(success=True, message_id=_message_id("suppressed"))

            task_turn = self._turns_by_task.get(asyncio.current_task())
            active_turn = task_turn or self._active_turns_by_chat.get(str(chat_id))
            if active_turn is not None:
                if active_turn.get("cancelled") or active_turn.get("turn_id") in self._cancelled_turns:
                    return SendResult(success=False, error="turn cancelled")
                progress = _parse_tool_progress(content)
                if progress:
                    message_id = _message_id("activity")
                    self._activity_groups_by_message[message_id] = progress
                    await self._emit_tool_progress(str(chat_id), active_turn["turn_id"], message_id, progress)
                    return SendResult(success=True, message_id=message_id)
                message_id = self._activity_message_by_turn.pop(active_turn["turn_id"], "") or _message_id("asst")
                payload = {
                    "message_id": message_id,
                    "role": "hermes",
                    "text": content if isinstance(content, str) else str(content),
                    "status": "running",
                    "created_at": _utc_iso(),
                    "turn_id": active_turn["turn_id"],
                }
                try:
                    event_kind = (
                        "message.updated"
                        if message_id in self._assistant_turn_by_message
                        else "message.created"
                    )
                    await self._append_chat_event_async(str(chat_id), event_kind, payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("vylen gateway: retained assistant send failed: %s", exc)
                    return SendResult(success=False, error=str(exc))
                self._assistant_turn_by_message[message_id] = active_turn["turn_id"]
                self._assistant_messages_by_turn.setdefault(active_turn["turn_id"], set()).add(message_id)
                return SendResult(success=True, message_id=message_id)

            # Plugin-initiated message — typically a Hermes cron with
            # `deliver=vylen`, or `BasePlatformAdapter.send_image` etc.
            # falling back to `.send`. We emit a `push` frame; the cloud
            # fans out to any SSE subscribers on /v1/notifications for the
            # owning user. No retries, no persistence — foreground pilot
            # only. Background delivery (FCM) ships behind the Firebase
            # Auth follow-up.
            if self._client is None:
                return SendResult(
                    success=False,
                    error="vylen gateway: socket not connected",
                    retryable=True,
                )
            raw = content if isinstance(content, str) else str(content)
            body, cron_job_id, cron_job_name = _parse_cron_envelope(raw)
            frame: dict[str, Any] = {
                "type": "push",
                "chat_id": _push_cursor_chat_id(chat_id),
                "text": body,
            }
            if cron_job_id:
                frame["cron_job_id"] = cron_job_id
            if cron_job_name:
                frame["cron_job_name"] = cron_job_name
            try:
                if self._chat_cursors is not None:
                    await self._chat_cursors.send_push(frame)
                else:
                    await self._client.send(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vylen gateway: push frame send failed: %s", exc)
                return SendResult(success=False, error=str(exc), retryable=True)
            return SendResult(success=True)

        async def edit_message(
            self,
            chat_id: str,
            message_id: str,
            content: str,
            *,
            finalize: bool = False,
        ):
            from gateway.platforms.base import SendResult

            progress = _parse_tool_progress(content)
            if progress or str(message_id) in self._activity_groups_by_message:
                task_turn = self._turns_by_task.get(asyncio.current_task())
                active_turn = task_turn or self._active_turns_by_chat.get(str(chat_id))
                turn_id = (
                    active_turn.get("turn_id")
                    if active_turn is not None
                    else self._assistant_turn_by_message.get(str(message_id), "")
                )
                if turn_id:
                    if turn_id in self._cancelled_turns:
                        return SendResult(success=False, error="turn cancelled")
                    self._activity_groups_by_message[str(message_id)] = progress
                    await self._emit_tool_progress(str(chat_id), turn_id, str(message_id), progress)
                    return SendResult(success=True, message_id=str(message_id))

            payload: dict[str, Any] = {
                "message_id": str(message_id),
                "text": content if isinstance(content, str) else str(content),
                "status": "completed" if finalize else "running",
                "updated_at": _utc_iso(),
            }
            turn_id = self._assistant_turn_by_message.get(str(message_id))
            if turn_id:
                if turn_id in self._cancelled_turns:
                    return SendResult(success=False, error="turn cancelled")
                payload["turn_id"] = turn_id
            try:
                await self._append_chat_event_async(str(chat_id), "message.updated", payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vylen gateway: retained assistant edit failed: %s", exc)
                return SendResult(success=False, error=str(exc))
            return SendResult(success=True, message_id=str(message_id))

        async def delete_message(self, chat_id: str, message_id: str) -> bool:
            payload: dict[str, Any] = {"message_id": str(message_id)}
            turn_id = self._assistant_turn_by_message.get(str(message_id))
            if turn_id:
                payload["turn_id"] = turn_id
            await self._append_chat_event_async(str(chat_id), "message.deleted", payload)
            return True

        async def on_processing_start(self, event) -> None:
            turn_id = _event_turn_id(event)
            if not turn_id:
                return
            chat_id = str(event.source.chat_id)
            user_message_id = str(getattr(event, "message_id", "") or "")
            active = self._active_turns_by_chat.get(chat_id)
            if active is None or active.get("turn_id") != turn_id:
                active = self._active_turn_for_event(event)
                self._active_turns_by_chat[chat_id] = active
            else:
                active["message_id"] = user_message_id
                active["source"] = event.source
            task = asyncio.current_task()
            if task is not None:
                self._turns_by_task[task] = {"chat_id": chat_id, **active}
            await self._append_chat_event_async(chat_id, "turn.started", {
                "turn_id": turn_id,
                "message_id": user_message_id,
                "started_at": _utc_iso(),
            })
            if user_message_id:
                await self._append_chat_event_async(chat_id, "message.updated", {
                    "message_id": user_message_id,
                    "turn_id": turn_id,
                    "status": "running",
                    "updated_at": _utc_iso(),
                })

        async def on_processing_complete(self, event, outcome) -> None:
            turn_id = _event_turn_id(event)
            if not turn_id:
                return
            chat_id = str(event.source.chat_id)
            user_message_id = str(getattr(event, "message_id", "") or "")
            task = asyncio.current_task()
            if task is not None:
                self._turns_by_task.pop(task, None)
            was_cancelled = turn_id in self._cancelled_turns
            outcome_value = str(getattr(outcome, "value", outcome) or "")
            if was_cancelled:
                self._append_cancel_marker_to_session_history(event.source, turn_id)
                self._cleanup_cancelled_turn(chat_id, turn_id)
                return
            if outcome_value == "cancelled":
                self._append_cancel_marker_to_session_history(event.source, turn_id)
                active = self._active_turns_by_chat.get(chat_id)
                if active and active.get("turn_id") == turn_id and active.get("cancel_requested"):
                    return
                kind = "turn.cancelled"
                payload = {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "reason": "cancelled",
                    "cancelled_at": _utc_iso(),
                }
                status = "cancelled"
            elif outcome_value == "failure":
                kind = "turn.failed"
                payload = {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "error": "message handler failed",
                    "failed_at": _utc_iso(),
                }
                status = "failed"
            else:
                kind = "turn.completed"
                payload = {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "completed_at": _utc_iso(),
                }
                status = "completed"
            await self._append_chat_event_async(chat_id, kind, payload)
            for activity_id in self._activity_ids_by_turn.pop(turn_id, set()):
                if self._activity_status_by_id.get(activity_id) in {"completed", "failed"}:
                    self._activity_payloads_by_id.pop(activity_id, None)
                    self._activity_status_by_id.pop(activity_id, None)
                    continue
                await self._append_activity_terminal(
                    chat_id,
                    turn_id,
                    activity_id,
                    status="failed" if status == "failed" else "completed",
                )
                self._activity_payloads_by_id.pop(activity_id, None)
                self._activity_status_by_id.pop(activity_id, None)
            for assistant_message_id in self._assistant_messages_by_turn.pop(turn_id, set()):
                await self._append_chat_event_async(chat_id, "message.updated", {
                    "message_id": assistant_message_id,
                    "turn_id": turn_id,
                    "status": status,
                    "updated_at": _utc_iso(),
                })
            if user_message_id:
                await self._append_chat_event_async(chat_id, "message.updated", {
                    "message_id": user_message_id,
                    "turn_id": turn_id,
                    "status": status,
                    "updated_at": _utc_iso(),
                })
            active = self._active_turns_by_chat.get(chat_id)
            if active and active.get("turn_id") == turn_id:
                self._active_turns_by_chat.pop(chat_id, None)
            self._promote_local_queued_event(event)
            if status == "completed":
                asyncio.create_task(self._sync_chat_title_from_hermes_later(chat_id, event.source))

        async def send_exec_approval(
            self,
            chat_id: str,
            command: str,
            session_key: str,
            description: str,
            metadata: Optional[dict[str, Any]] = None,
        ):
            from gateway.platforms.base import SendResult

            return await self._create_action_card(
                chat_id=chat_id,
                kind="approval",
                event_kind="approval.requested",
                session_key=session_key,
                payload={
                    "command": command,
                    "description": description,
                    "choices": ["once", "session", "always", "deny"],
                },
                message_prefix="approval",
            )

        async def send_slash_confirm(
            self,
            chat_id: str,
            title: str,
            message: str,
            session_key: str,
            confirm_id: str,
            metadata: Optional[dict[str, Any]] = None,
        ):
            if self._consume_native_confirm_session(session_key, confirm_id):
                from gateway.platforms.base import SendResult
                from tools import slash_confirm as _slash_confirm_mod

                await _slash_confirm_mod.resolve(
                    session_key,
                    confirm_id,
                    "once",
                    timeout=self._action_ttl_seconds,
                )
                return SendResult(success=True, message_id=_message_id("confirm"))
            return await self._create_action_card(
                chat_id=chat_id,
                kind="confirm",
                event_kind="confirm.requested",
                session_key=session_key,
                payload={
                    "title": title,
                    "message": message,
                    "choices": ["once", "always", "cancel"],
                    "confirm_id": confirm_id,
                },
                message_prefix="confirm",
                action_id=confirm_id,
            )

        async def send_image_file(
            self,
            chat_id: str,
            image_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
            **kwargs,
        ):
            # Tunnel-streamed media: register the local file in the
            # BlobRegistry, get a short-lived token, ship just that on the
            # push frame. The client fetches the bytes via
            # /v1/instances/<id>/blobs/<token> on the cloud, which tunnels
            # the read through the existing gateway WS (see relay._serve_blob).
            # Push frames stay small; the SSE channel doesn't carry image
            # bytes; multiple clients reuse the same URL.
            from gateway.platforms.base import SendResult
            if self._client is None or self._blobs is None:
                return SendResult(
                    success=False,
                    error="vylen gateway: socket not connected",
                    retryable=True,
                )
            registered = await self._blobs.register(image_path)
            if registered is None:
                logger.warning("vylen gateway: could not register image %s", image_path)
                # Fall back to the base class behaviour (path-as-text) so the
                # user at least sees that an image was attempted, rather than
                # silent dropping.
                return await super(VylenGatewayAdapter, self).send_image_file(
                    chat_id=chat_id,
                    image_path=image_path,
                    caption=caption,
                    reply_to=reply_to,
                    metadata=metadata,
                    **kwargs,
                )
            token, mime, filename = registered
            task_turn = self._turns_by_task.get(asyncio.current_task())
            active_turn = task_turn or self._active_turns_by_chat.get(str(chat_id))
            if active_turn is not None:
                turn_id = str(active_turn.get("turn_id") or "")
                if active_turn.get("cancelled") or turn_id in self._cancelled_turns:
                    return SendResult(success=False, error="turn cancelled")
                if not self._instance_id:
                    return SendResult(success=False, error="vylen gateway: instance id unavailable")
                message_id = _message_id("asst")
                payload = {
                    "message_id": message_id,
                    "role": "hermes",
                    "text": caption or "",
                    "status": "running",
                    "created_at": _utc_iso(),
                    "turn_id": turn_id,
                    "attachments": [{
                        "type": "image",
                        "data_url": f"/v1/instances/{self._instance_id}/blobs/{token}",
                        "mime_type": mime,
                        "filename": filename,
                    }],
                }
                try:
                    await self._append_chat_event_async(str(chat_id), "message.created", payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("vylen gateway: retained image message failed: %s", exc)
                    return SendResult(success=False, error=str(exc))
                self._assistant_turn_by_message[message_id] = turn_id
                self._assistant_messages_by_turn.setdefault(turn_id, set()).add(message_id)
                return SendResult(success=True, message_id=message_id)

            raw_caption = caption or ""
            body, cron_job_id, cron_job_name = _parse_cron_envelope(raw_caption)
            frame: dict[str, Any] = {
                "type": "push",
                "chat_id": _push_cursor_chat_id(chat_id),
                "text": body,
                "image_token": token,
                "image_mime": mime,
                "image_filename": filename,
            }
            if cron_job_id:
                frame["cron_job_id"] = cron_job_id
            if cron_job_name:
                frame["cron_job_name"] = cron_job_name
            try:
                if self._chat_cursors is not None:
                    await self._chat_cursors.send_push(frame)
                else:
                    await self._client.send(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vylen gateway: image push frame send failed: %s", exc)
                return SendResult(success=False, error=str(exc), retryable=True)
            return SendResult(success=True)

        async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
            return {"name": "vylen", "type": "dm"}

        async def _create_action_card(
            self,
            *,
            chat_id: str,
            kind: str,
            event_kind: str,
            session_key: str,
            payload: dict[str, Any],
            message_prefix: str,
            action_id: str | None = None,
            record_extra: dict[str, Any] | None = None,
        ):
            from gateway.platforms.base import SendResult

            active_turn = self._active_turns_by_chat.get(str(chat_id)) or {}
            turn_id = active_turn.get("turn_id")
            message_id = _message_id(message_prefix)
            resolved_action_id = action_id or f"{kind}_{uuid.uuid4().hex}"
            now = time.time()
            expires_at = now + self._action_ttl_seconds
            record = {
                "kind": kind,
                "chat_id": str(chat_id),
                "turn_id": turn_id,
                "message_id": message_id,
                "session_key": session_key,
                "expires_at": expires_at,
                "expired_emitted": False,
            }
            if record_extra:
                record.update(record_extra)
            self._action_cards[resolved_action_id] = record
            self._action_cards.move_to_end(resolved_action_id)
            while len(self._action_cards) > 1000:
                self._action_cards.popitem(last=False)
            retained_payload = {
                **payload,
                "turn_id": turn_id,
                "action_id": resolved_action_id,
                "message_id": message_id,
                "created_at": _utc_iso(now),
                "expires_at": _utc_iso(expires_at),
            }
            if kind == "confirm" and not retained_payload.get("confirm_id"):
                retained_payload["confirm_id"] = resolved_action_id
            await self._append_chat_event_async(str(chat_id), event_kind, retained_payload)
            return SendResult(success=True, message_id=message_id)

        async def _handle_chat_message(self, frame: dict[str, Any]) -> None:
            request_id = _safe_id(frame.get("request_id"))
            chat_id = _safe_id(frame.get("chat_id"))
            client_message_id = _safe_id(frame.get("client_message_id"))
            user_id = _safe_id(frame.get("user_id"))
            if not request_id or not chat_id or not client_message_id or not user_id:
                await self._send_chat_message_error(frame, "CHAT_MESSAGE_INVALID", "chat_message is missing required ids")
                return

            dedup_key = (chat_id, client_message_id)
            existing = await self._chat_dedup_lookup_async(chat_id, client_message_id)
            if existing is not None:
                await self._send_frame({
                    "type": FRAME_CHAT_MESSAGE_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "client_message_id": client_message_id,
                    "message_id": existing.get("user_message_id"),
                    "turn_id": existing["turn_id"],
                    "accepted": True,
                })
                return

            try:
                attachment_result = self._decode_attachments(frame.get("attachments"))
                await self._register_decoded_attachment_urls(attachment_result)
            except ValueError as exc:
                await self._send_chat_message_error(frame, "INVALID_ATTACHMENT", str(exc))
                return

            text = frame.get("text")
            text = text if isinstance(text, str) else ""
            attachments = attachment_result["public"]
            if not attachments and _native_chat_command(text):
                await self._handle_native_chat_command(
                    frame=frame,
                    chat_id=chat_id,
                    client_message_id=client_message_id,
                    text=text.strip(),
                    dedup_key=dedup_key,
                    request_id=request_id,
                )
                return
            turn_id = f"turn_{uuid.uuid4().hex}"
            user_message_id = _message_id("user")
            event = self._build_message_event(
                frame=frame,
                chat_id=chat_id,
                text=text,
                user_message_id=user_message_id,
                turn_id=turn_id,
                media_urls=attachment_result["paths"],
                media_types=attachment_result["types"],
            )
            queue_instead_of_interrupt = self._should_queue_new_message(chat_id, event)
            accepted = {
                "turn_id": turn_id,
                "user_message_id": user_message_id,
                "accepted_at": time.time(),
                "source": event.source,
            }
            author = {
                "id": user_id,
                "name": str(frame.get("user_name") or user_id),
            }
            try:
                await self._append_chat_event_async(chat_id, "message.created", {
                    "message_id": user_message_id,
                    "role": "user",
                    "text": text,
                    "status": "queued" if queue_instead_of_interrupt else "running",
                    "created_at": _utc_iso(),
                    "turn_id": turn_id,
                    "client_message_id": client_message_id,
                    "origin_client_id": str(frame.get("client_id") or ""),
                    "chat_name": str(frame.get("chat_name") or ""),
                    "author": author,
                    "attachments": attachments,
                })
                await self._chat_dedup_record_async(chat_id, client_message_id, accepted)
            except Exception as exc:  # noqa: BLE001
                await self._chat_dedup_forget_async(chat_id, client_message_id)
                await self._send_chat_message_error(frame, _chat_state_error_code(exc), str(exc))
                return

            if queue_instead_of_interrupt:
                await self._append_chat_event_async(chat_id, "turn.queued", {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "queued_at": _utc_iso(),
                })
                self._enqueue_message_event(event)
                await self._append_session_status(chat_id, source=event.source)
            else:
                self._active_turns_by_chat[chat_id] = self._active_turn_for_event(event)

            await self._send_frame({
                "type": FRAME_CHAT_MESSAGE_ACK,
                "request_id": request_id,
                "chat_id": chat_id,
                "client_message_id": client_message_id,
                "message_id": user_message_id,
                "turn_id": turn_id,
                "accepted": True,
            })

            if queue_instead_of_interrupt:
                return

            task = asyncio.create_task(self._process_chat_message(chat_id, user_message_id, turn_id, event))
            self._chat_message_tasks.add(task)
            task.add_done_callback(self._chat_message_tasks.discard)
            await asyncio.sleep(0.1)

        async def _process_chat_message(self, chat_id: str, user_message_id: str, turn_id: str, event: Any) -> None:
            if turn_id in self._cancelled_turns:
                self._cleanup_cancelled_turn(chat_id, turn_id)
                return
            try:
                await self.handle_message(event)
            except Exception as exc:  # noqa: BLE001
                await self._append_chat_event_async(chat_id, "turn.failed", {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "error": str(exc),
                    "failed_at": _utc_iso(),
                })
                await self._append_chat_event_async(chat_id, "message.updated", {
                    "message_id": user_message_id,
                    "turn_id": turn_id,
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": _utc_iso(),
                })

        async def _handle_native_chat_command(
            self,
            *,
            frame: dict[str, Any],
            chat_id: str,
            client_message_id: str,
            text: str,
            dedup_key: tuple[str, str],
            request_id: str,
        ) -> None:
            turn_id = _message_id("command_turn")
            source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
            command, args = _split_native_chat_command(text)
            accepted = {
                "turn_id": turn_id,
                "user_message_id": _message_id("command"),
                "accepted_at": time.time(),
            }
            if source is not None:
                accepted["source"] = source

            if command == "status":
                try:
                    await self._chat_dedup_record_async(chat_id, client_message_id, accepted)
                    await self._append_session_status(chat_id, source=source)
                    await self._dispatch_native_command(frame, chat_id, "/status", suppress_confirm=False)
                except Exception as exc:  # noqa: BLE001
                    await self._chat_dedup_forget_async(chat_id, client_message_id)
                    await self._send_chat_message_error(frame, _chat_state_error_code(exc, fallback="SESSION_STATUS_FAILED"), str(exc))
                    return
            elif command == "reset":
                active = self._active_turns_by_chat.get(chat_id)
                if active is not None:
                    await self._chat_dedup_forget_async(chat_id, client_message_id)
                    await self._send_chat_message_error(frame, "TURN_ACTIVE", "Stop the current run first")
                    return
                if source is None:
                    await self._chat_dedup_forget_async(chat_id, client_message_id)
                    await self._send_chat_message_error(frame, "SESSION_SOURCE_UNAVAILABLE", "Could not route session reset")
                    return
                try:
                    await self._chat_dedup_record_async(chat_id, client_message_id, accepted)
                    await self._create_action_card(
                        chat_id=chat_id,
                        kind="confirm",
                        event_kind="confirm.requested",
                        session_key=self._session_key_for_source(source),
                        payload={
                            "title": "Reset chat history",
                            "message": "Hermes will forget the conversation above. Your messages stay visible to you.",
                            "choices": ["once", "cancel"],
                            "confirm_id": "",
                        },
                        message_prefix="confirm",
                        record_extra={
                            "native_action": "session.reset",
                            "source": source,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._chat_dedup_forget_async(chat_id, client_message_id)
                    await self._send_chat_message_error(frame, _chat_state_error_code(exc), str(exc))
                    return
            elif command == "queue":
                await self._chat_dedup_forget_async(chat_id, client_message_id)
                frame = dict(frame)
                frame["text"] = args
                await self._handle_message_control_action(
                    frame,
                    chat_id=chat_id,
                    request_id=request_id,
                    client_message_id=client_message_id,
                    control="queue",
                    ack_type=FRAME_CHAT_MESSAGE_ACK,
                )
                return
            elif command == "steer":
                await self._chat_dedup_forget_async(chat_id, client_message_id)
                frame = dict(frame)
                frame["text"] = args
                await self._handle_message_control_action(
                    frame,
                    chat_id=chat_id,
                    request_id=request_id,
                    client_message_id=client_message_id,
                    control="steer",
                    ack_type=FRAME_CHAT_MESSAGE_ACK,
                )
                return
            else:
                await self._chat_dedup_forget_async(chat_id, client_message_id)
                await self._send_chat_message_error(frame, "CHAT_MESSAGE_INVALID", "unsupported native command")
                return

            await self._send_frame({
                "type": FRAME_CHAT_MESSAGE_ACK,
                "request_id": request_id,
                "chat_id": chat_id,
                "client_message_id": client_message_id,
                "turn_id": turn_id,
                "accepted": True,
            })

        async def _handle_message_control_action(
            self,
            frame: dict[str, Any],
            *,
            chat_id: str,
            request_id: str,
            client_message_id: str,
            control: str,
            ack_type: str,
        ) -> None:
            dedup_key = (chat_id, client_message_id)
            existing = await self._chat_dedup_lookup_async(chat_id, client_message_id)
            if existing is not None:
                await self._send_frame({
                    "type": ack_type,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "client_message_id": client_message_id,
                    "message_id": existing.get("user_message_id"),
                    "turn_id": existing.get("turn_id"),
                    "intent": existing.get("intent"),
                    "message_status": existing.get("message_status"),
                    "accepted": True,
                })
                return

            text = frame.get("text")
            text = text if isinstance(text, str) else ""
            text = text.strip()
            try:
                attachment_result = self._decode_attachments(frame.get("attachments"))
                await self._register_decoded_attachment_urls(attachment_result)
            except ValueError as exc:
                await self._send_message_control_error(frame, ack_type, "INVALID_ATTACHMENT", str(exc))
                return
            if not text and not attachment_result["public"]:
                await self._send_message_control_error(frame, ack_type, "CHAT_MESSAGE_INVALID", "text is required")
                return
            if control == "steer" and attachment_result["public"]:
                await self._send_message_control_error(frame, ack_type, "STEER_ATTACHMENTS_UNSUPPORTED", "Steer only supports text")
                return

            user_message_id = _message_id("user")
            turn_id = f"turn_{uuid.uuid4().hex}"
            source = self._source_for_chat_action(frame, chat_id, message_id=user_message_id)
            if source is None:
                await self._send_message_control_error(
                    frame,
                    ack_type,
                    "SESSION_SOURCE_UNAVAILABLE",
                    "Could not route message",
                )
                return
            event = self._build_message_event(
                frame=frame,
                chat_id=chat_id,
                text=text,
                user_message_id=user_message_id,
                turn_id=turn_id,
                media_urls=attachment_result["paths"],
                media_types=attachment_result["types"],
            )
            if control == "steer" and not self._should_queue_new_message(chat_id, event):
                control = "queue"
            queued = control == "queue" and self._should_queue_new_message(chat_id, event)
            status = "queued" if queued else ("completed" if control == "steer" else "running")
            author = {
                "id": str(frame.get("user_id") or ""),
                "name": str(frame.get("user_name") or frame.get("user_id") or ""),
            }
            try:
                await self._append_chat_event_async(chat_id, "message.created", {
                    "message_id": user_message_id,
                    "role": "user",
                    "text": text,
                    "status": status,
                    "created_at": _utc_iso(),
                    "turn_id": turn_id,
                    "client_message_id": client_message_id,
                    "origin_client_id": str(frame.get("client_id") or ""),
                    "author": author,
                    "attachments": attachment_result["public"],
                    "intent": control,
                })
            except Exception as exc:  # noqa: BLE001
                await self._send_message_control_error(frame, ack_type, _chat_state_error_code(exc), str(exc))
                return
            if control == "steer":
                fallback_pending = False
                session_key = self._session_key_for_source(source)
                pending_before = getattr(self, "_pending_messages", {}).get(session_key)
                try:
                    dispatched = await self._dispatch_native_command(
                        frame,
                        chat_id,
                        f"/steer {text}",
                        suppress_confirm=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._append_chat_event_async(chat_id, "message.updated", {
                        "message_id": user_message_id,
                        "turn_id": turn_id,
                        "status": "failed",
                        "error": str(exc),
                        "updated_at": _utc_iso(),
                    })
                    await self._send_message_control_error(frame, ack_type, "STEER_FAILED", str(exc))
                    return
                if not dispatched:
                    await self._append_chat_event_async(chat_id, "message.updated", {
                        "message_id": user_message_id,
                        "turn_id": turn_id,
                        "status": "failed",
                        "error": "Could not route steer",
                        "updated_at": _utc_iso(),
                    })
                    await self._send_message_control_error(
                        frame,
                        ack_type,
                        "SESSION_SOURCE_UNAVAILABLE",
                        "Could not route steer",
                    )
                    return
                pending = getattr(self, "_pending_messages", {}).get(session_key)
                if (
                    pending is not None
                    and pending is not pending_before
                    and getattr(pending, "message_id", None) != user_message_id
                ):
                    pending_text = str(getattr(pending, "text", "") or "").strip()
                    if pending_text == text:
                        if pending_before is None:
                            self._pending_messages.pop(session_key, None)
                        else:
                            self._pending_messages[session_key] = pending_before
                        self._enqueue_message_event(event)
                        fallback_pending = True
                if fallback_pending:
                    control = "queue"
                    status = "queued"
                    await self._append_chat_event_async(chat_id, "message.updated", {
                        "message_id": user_message_id,
                        "turn_id": turn_id,
                        "status": "queued",
                        "intent": "queue",
                        "updated_at": _utc_iso(),
                    })
                    await self._append_chat_event_async(chat_id, "turn.queued", {
                        "turn_id": turn_id,
                        "message_id": user_message_id,
                        "queued_at": _utc_iso(),
                    })
                    await self._append_session_status(chat_id, source=event.source)
            elif queued:
                await self._append_chat_event_async(chat_id, "turn.queued", {
                    "turn_id": turn_id,
                    "message_id": user_message_id,
                    "queued_at": _utc_iso(),
                })
                self._enqueue_message_event(event)
                await self._append_session_status(chat_id, source=event.source)
            else:
                self._active_turns_by_chat[chat_id] = self._active_turn_for_event(event)
                task = asyncio.create_task(self._process_chat_message(chat_id, user_message_id, turn_id, event))
                self._chat_message_tasks.add(task)
                task.add_done_callback(self._chat_message_tasks.discard)

            accepted = {
                "turn_id": turn_id,
                "user_message_id": user_message_id,
                "accepted_at": time.time(),
                "source": event.source,
                "intent": control,
                "message_status": status,
            }
            try:
                await self._chat_dedup_record_async(chat_id, client_message_id, accepted)
            except Exception as exc:  # noqa: BLE001
                await self._send_message_control_error(frame, ack_type, _chat_state_error_code(exc), str(exc))
                return

            await self._send_frame({
                "type": ack_type,
                "request_id": request_id,
                "chat_id": chat_id,
                "client_message_id": client_message_id,
                "message_id": user_message_id,
                "turn_id": turn_id,
                "intent": control,
                "message_status": status,
                "accepted": True,
            })
            await asyncio.sleep(0.1)

        async def _handle_chat_action(self, frame: dict[str, Any]) -> None:
            request_id = _safe_id(frame.get("request_id"))
            chat_id = _safe_id(frame.get("chat_id"))
            action = str(frame.get("action") or "")
            action_id = _safe_id(frame.get("action_id"))
            if not request_id or not chat_id or not action:
                await self._send_chat_action_error(frame, "CHAT_ACTION_INVALID", "chat_action is missing required ids")
                return
            if action == "turn.cancel":
                turn_id = _safe_id(frame.get("turn_id"))
                active = self._active_turns_by_chat.get(chat_id)
                if not turn_id or active is None or active.get("turn_id") != turn_id:
                    await self._send_chat_action_error(frame, "TURN_NOT_ACTIVE", "This turn is no longer active")
                    return
                active["cancel_requested"] = True
                try:
                    await self._dispatch_native_stop(chat_id, active)
                except Exception as exc:  # noqa: BLE001
                    active.pop("cancel_requested", None)
                    await self._send_chat_action_error(frame, "TURN_CANCEL_FAILED", str(exc))
                    return
                await self._cancel_active_turn(chat_id, active, reason="user_stop")
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "turn_id": turn_id,
                    "accepted": True,
                })
                return
            if action == "chat.delete":
                try:
                    if hasattr(self._chat_event_logs, "mark_deleted"):
                        await asyncio.to_thread(self._chat_event_logs.mark_deleted, chat_id)
                    else:
                        await self._append_chat_event_async(chat_id, "chat.deleted", {
                            "chat_id": chat_id,
                            "deleted_at": _utc_iso(),
                        })
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, _chat_state_error_code(exc), str(exc))
                    return
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                })
                return
            if action == "chat.rename":
                title = str(frame.get("title") or frame.get("chat_name") or frame.get("text") or "").strip()
                if not title:
                    await self._send_chat_action_error(frame, "CHAT_RENAME_INVALID", "title is required")
                    return
                try:
                    if hasattr(self._chat_event_logs, "rename_chat"):
                        event = await asyncio.to_thread(self._chat_event_logs.rename_chat, chat_id, title)
                        title = str(event.payload.get("title") or title)
                    else:
                        await self._append_chat_event_async(chat_id, "chat.renamed", {
                            "chat_id": chat_id,
                            "title": title,
                            "renamed_at": _utc_iso(),
                        })
                    await self._sync_hermes_session_title(frame, chat_id, title)
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, _chat_state_error_code(exc), str(exc))
                    return
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                    "title": title,
                })
                return
            if action == "session.status":
                source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
                try:
                    await self._append_session_status(chat_id, source=source)
                    await self._dispatch_native_command(frame, chat_id, "/status", suppress_confirm=False)
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, _chat_state_error_code(exc, fallback="SESSION_STATUS_FAILED"), str(exc))
                    return
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                })
                return
            if action == "session.controls":
                source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
                try:
                    await self._append_session_controls(chat_id, source=source)
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, _chat_state_error_code(exc), str(exc))
                    return
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                })
                return
            if action == "session.reasoning":
                value = str(frame.get("text") or "").strip().lower()
                if value not in {"none", "minimal", "low", "medium", "high", "xhigh", "reset", "show", "hide", "on", "off"}:
                    await self._send_chat_action_error(frame, "SESSION_REASONING_INVALID", "Unsupported reasoning value")
                    return
                command_value = "show" if value == "on" else "hide" if value == "off" else value
                source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
                session_key = self._session_key_for_source(source) if source is not None else ""
                wait_for_reasoning = not self._session_task_running(session_key)
                try:
                    dispatched = await self._dispatch_native_command(
                        frame,
                        chat_id,
                        f"/reasoning {command_value}",
                        suppress_confirm=False,
                        wait_for_completion=wait_for_reasoning,
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, "SESSION_REASONING_FAILED", str(exc))
                    return
                if not dispatched:
                    await self._send_chat_action_error(frame, "SESSION_SOURCE_UNAVAILABLE", "Could not route reasoning command")
                    return
                try:
                    await self._append_session_controls(chat_id, source=source)
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, _chat_state_error_code(exc), str(exc))
                    return
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                })
                return
            if action == "session.reset":
                active = self._active_turns_by_chat.get(chat_id)
                if active is not None:
                    await self._send_chat_action_error(frame, "TURN_ACTIVE", "Stop the current run first")
                    return
                try:
                    dispatched = await self._dispatch_native_command(
                        frame,
                        chat_id,
                        "/reset",
                        suppress_confirm=True,
                        expected_confirm_commands={"new"},
                        wait_for_completion=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._send_chat_action_error(frame, "SESSION_RESET_FAILED", str(exc))
                    return
                if not dispatched:
                    await self._send_chat_action_error(frame, "SESSION_SOURCE_UNAVAILABLE", "Could not route session reset")
                    return
                await self._append_chat_event_async(chat_id, "session.reset", {
                    "message_id": _message_id("reset"),
                    "text": "Cleared history",
                    "cleared_at": _utc_iso(),
                })
                source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
                await self._append_session_status(chat_id, source=source)
                await self._send_frame({
                    "type": FRAME_CHAT_ACTION_ACK,
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "accepted": True,
                })
                return
            if action in {"message.queue", "message.steer"}:
                client_message_id = _safe_id(frame.get("client_message_id"))
                if not client_message_id:
                    await self._send_chat_action_error(frame, "CHAT_ACTION_INVALID", "client_message_id is required")
                    return
                await self._handle_message_control_action(
                    frame,
                    chat_id=chat_id,
                    request_id=request_id,
                    client_message_id=client_message_id,
                    control="queue" if action == "message.queue" else "steer",
                    ack_type=FRAME_CHAT_ACTION_ACK,
                )
                return
            if action not in {"approval.respond", "confirm.respond"} or not action_id:
                await self._send_chat_action_error(frame, "CHAT_ACTION_INVALID", "unsupported chat action")
                return

            record = self._action_cards.get(action_id)
            expected_kind = "approval" if action == "approval.respond" else "confirm"
            if record is None or record.get("chat_id") != chat_id or record.get("kind") != expected_kind:
                await self._send_chat_action_error(frame, "STALE_ACTION", "This action is no longer available")
                return
            if float(record.get("expires_at") or 0) <= time.time():
                await self._emit_action_expired(chat_id, action_id, expected_kind, record, reason="stale_action")
                await self._send_chat_action_error(frame, "STALE_ACTION", "This action is no longer available")
                return

            choice = str(frame.get("choice") or "")
            if expected_kind == "approval" and choice not in {"once", "session", "always", "deny"}:
                await self._send_chat_action_error(frame, "INVALID_CHOICE", "Invalid approval choice")
                return
            if expected_kind == "confirm" and choice not in {"once", "always", "cancel"}:
                await self._send_chat_action_error(frame, "INVALID_CHOICE", "Invalid confirmation choice")
                return

            try:
                # Approval and slash-confirm cards are Hermes callback
                # surfaces. Resolve the pending callback; do not mutate
                # Hermes running-agent/session internals from this branch.
                if expected_kind == "approval":
                    from tools.approval import resolve_gateway_approval

                    resolved_count = resolve_gateway_approval(record["session_key"], choice)
                    if not resolved_count:
                        await self._emit_action_expired(chat_id, action_id, expected_kind, record, reason="resolver_lost")
                        await self._send_chat_action_error(frame, "STALE_ACTION", "This approval is no longer available")
                        return
                elif record.get("native_action") == "session.reset":
                    if choice != "cancel":
                        active = self._active_turns_by_chat.get(chat_id)
                        if active is not None:
                            await self._send_chat_action_error(frame, "TURN_ACTIVE", "Stop the current run first")
                            return
                        dispatched = await self._dispatch_native_command(
                            frame,
                            chat_id,
                            "/reset",
                            suppress_confirm=True,
                            expected_confirm_commands={"new"},
                            wait_for_completion=True,
                        )
                        if not dispatched:
                            await self._send_chat_action_error(
                                frame,
                                "SESSION_SOURCE_UNAVAILABLE",
                                "Could not route session reset",
                            )
                            return
                        await self._append_chat_event_async(chat_id, "session.reset", {
                            "message_id": _message_id("reset"),
                            "text": "Cleared history",
                            "cleared_at": _utc_iso(),
                        })
                        source = record.get("source")
                        await self._append_session_status(chat_id, source=source)
                else:
                    from tools import slash_confirm as _slash_confirm_mod

                    pending = _slash_confirm_mod.get_pending(record["session_key"])
                    if not pending or pending.get("confirm_id") != action_id:
                        await self._emit_action_expired(chat_id, action_id, expected_kind, record, reason="resolver_lost")
                        await self._send_chat_action_error(frame, "STALE_ACTION", "This confirmation is no longer available")
                        return
                    follow_up = await _slash_confirm_mod.resolve(
                        record["session_key"],
                        action_id,
                        choice,
                        timeout=self._action_ttl_seconds,
                    )
                    if follow_up:
                        await self._append_assistant_message(chat_id, follow_up, record.get("turn_id"))
            except Exception as exc:  # noqa: BLE001
                await self._send_chat_action_error(frame, "ACTION_RESOLVE_FAILED", str(exc))
                return

            self._action_cards.pop(action_id, None)
            resolved_kind = "approval.resolved" if expected_kind == "approval" else "confirm.resolved"
            payload = {
                "turn_id": record.get("turn_id"),
                "action_id": action_id,
                "message_id": record.get("message_id"),
                "choice": choice,
                "updated_at": _utc_iso(),
            }
            if expected_kind == "approval":
                payload["resolved"] = choice != "deny"
            await self._append_chat_event_async(chat_id, resolved_kind, payload)
            await self._send_frame({
                "type": FRAME_CHAT_ACTION_ACK,
                "request_id": request_id,
                "chat_id": chat_id,
                "action_id": action_id,
                "accepted": True,
            })

        def _build_message_event(
            self,
            *,
            frame: dict[str, Any],
            chat_id: str,
            text: str,
            user_message_id: str,
            turn_id: str,
            media_urls: list[str],
            media_types: list[str],
        ):
            from gateway.platforms.base import MessageEvent, MessageType
            from gateway.session import SessionSource

            message_type = MessageType.TEXT
            if media_types:
                first_type = media_types[0]
                if first_type == "image":
                    message_type = MessageType.PHOTO
                elif first_type == "voice":
                    message_type = MessageType.VOICE
                elif first_type == "audio":
                    message_type = MessageType.AUDIO
            source = SessionSource(
                platform=self.platform,
                chat_id=chat_id,
                chat_name=str(frame.get("chat_name") or chat_id),
                chat_type="dm",
                user_id=str(frame.get("user_id") or ""),
                user_name=str(frame.get("user_name") or frame.get("user_id") or ""),
                message_id=user_message_id,
            )
            raw_message = dict(frame)
            raw_message["turn_id"] = turn_id
            raw_message["user_message_id"] = user_message_id
            return MessageEvent(
                text=text,
                message_type=message_type,
                source=source,
                raw_message=raw_message,
                message_id=user_message_id,
                media_urls=media_urls,
                media_types=media_types,
            )

        def _build_command_event(self, *, source: Any, text: str, turn_id: str):
            from gateway.platforms.base import MessageEvent, MessageType

            raw_message = {
                "text": text,
                "turn_id": turn_id,
                "native_action": True,
            }
            return MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=raw_message,
                message_id=_message_id("command"),
            )

        async def _register_decoded_attachment_urls(self, attachment_result: dict[str, list[Any]]) -> None:
            public = attachment_result.get("public") or []
            paths = attachment_result.get("paths") or []
            if not public:
                return
            if self._blobs is None or not self._instance_id:
                raise ValueError("attachment blob registry is unavailable")
            for attachment, path in zip(public, paths, strict=False):
                registered = await self._blobs.register(path)
                if registered is None:
                    raise ValueError("attachment file is unavailable")
                token, mime_type, filename = registered
                attachment["data_url"] = f"/v1/instances/{self._instance_id}/blobs/{token}"
                attachment["mime_type"] = mime_type
                attachment["filename"] = attachment.get("filename") or filename

        def _decode_attachments(self, attachments: Any) -> dict[str, list[Any]]:
            if attachments is None:
                return {"paths": [], "types": [], "public": []}
            if not isinstance(attachments, list):
                raise ValueError("attachments must be an array")
            paths: list[str] = []
            media_types: list[str] = []
            public: list[dict[str, Any]] = []
            total = 0
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    raise ValueError("attachment must be an object")
                data_url = attachment.get("data_url")
                if not isinstance(data_url, str) or not data_url:
                    raise ValueError("attachment data_url is required")
                mime_type, data = _decode_data_url(data_url)
                declared_mime = str(attachment.get("mime_type") or mime_type).strip().lower()
                if declared_mime and declared_mime != mime_type:
                    raise ValueError("attachment mime_type does not match data_url")
                total += len(data)
                if len(data) > self._attachment_max_bytes or total > self._attachment_total_max_bytes:
                    raise ValueError("attachment is too large")
                ext = mimetypes.guess_extension(mime_type) or ""
                if mime_type.startswith("image/"):
                    from gateway.platforms.base import cache_image_from_bytes

                    path = cache_image_from_bytes(data, ext or ".jpg")
                    media_type = "image"
                elif mime_type.startswith("audio/"):
                    from gateway.platforms.base import cache_audio_from_bytes

                    path = cache_audio_from_bytes(data, ext or ".ogg")
                    media_type = "voice" if str(attachment.get("type") or "") == "voice" else "audio"
                else:
                    raise ValueError("Only image and audio attachments are supported")
                paths.append(path)
                media_types.append(media_type)
                public.append({
                    "id": str(attachment.get("id") or ""),
                    "type": str(attachment.get("type") or media_type),
                    "mime_type": mime_type,
                    "filename": str(attachment.get("filename") or os.path.basename(path)),
                })
            return {"paths": paths, "types": media_types, "public": public}

        async def _append_assistant_message(self, chat_id: str, text: str, turn_id: str | None) -> str:
            message_id = _message_id("asst")
            payload: dict[str, Any] = {
                "message_id": message_id,
                "role": "hermes",
                "text": text,
                "status": "completed",
                "created_at": _utc_iso(),
            }
            if turn_id:
                payload["turn_id"] = turn_id
                self._assistant_turn_by_message[message_id] = turn_id
                self._assistant_messages_by_turn.setdefault(turn_id, set()).add(message_id)
            await self._append_chat_event_async(chat_id, "message.created", payload)
            return message_id

        def _append_chat_event_sync(self, chat_id: str, kind: str, payload: dict[str, Any]) -> int | None:
            if self._chat_cursors is not None:
                seq = self._chat_cursors.append_event(chat_id, kind, payload)
                self._maybe_schedule_chat_state_sweep()
                return seq
            event = self._chat_event_logs.get_or_create(chat_id).append(kind, payload)
            self._maybe_schedule_chat_state_sweep()
            return event.seq

        def _append_chat_event(self, chat_id: str, kind: str, payload: dict[str, Any]) -> int | None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return self._append_chat_event_sync(chat_id, kind, payload)
            asyncio.create_task(self._append_chat_event_async(chat_id, kind, payload))
            return None

        async def _append_chat_event_async(self, chat_id: str, kind: str, payload: dict[str, Any]) -> int | None:
            seq = await asyncio.to_thread(self._append_chat_event_sync, chat_id, kind, payload)
            self._maybe_schedule_chat_state_sweep()
            return seq

        def _maybe_schedule_chat_state_sweep(self) -> None:
            if not hasattr(self._chat_event_logs, "consume_sweep_requested"):
                return
            try:
                requested = self._chat_event_logs.consume_sweep_requested()
            except Exception:  # noqa: BLE001
                return
            if not requested:
                return
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return
            if self._chat_sweep_request_task is not None and not self._chat_sweep_request_task.done():
                return
            self._chat_sweep_request_task = asyncio.create_task(
                _run_single_sweep(self._chat_event_logs, label="chat event log")
            )

        def _chat_dedup_lookup(self, chat_id: str, client_message_id: str) -> dict[str, Any] | None:
            dedup_key = (chat_id, client_message_id)
            existing = self._accepted_chat_messages.get(dedup_key)
            if existing is not None:
                self._accepted_chat_messages.move_to_end(dedup_key)
                return existing
            if hasattr(self._chat_event_logs, "dedup_lookup"):
                try:
                    return self._chat_event_logs.dedup_lookup(chat_id, client_message_id)
                except Exception:  # noqa: BLE001
                    return None
            return None

        async def _chat_dedup_lookup_async(self, chat_id: str, client_message_id: str) -> dict[str, Any] | None:
            return await asyncio.to_thread(self._chat_dedup_lookup, chat_id, client_message_id)

        def _chat_dedup_record(self, chat_id: str, client_message_id: str, accepted: dict[str, Any]) -> None:
            dedup_key = (chat_id, client_message_id)
            if hasattr(self._chat_event_logs, "dedup_record"):
                self._chat_event_logs.dedup_record(
                    chat_id,
                    client_message_id,
                    turn_id=str(accepted.get("turn_id") or ""),
                    message_id=str(accepted.get("user_message_id") or ""),
                    payload={
                        "turn_id": str(accepted.get("turn_id") or ""),
                        "user_message_id": str(accepted.get("user_message_id") or ""),
                        "intent": str(accepted.get("intent") or ""),
                        "message_status": str(accepted.get("message_status") or ""),
                    },
                )
            self._accepted_chat_messages[dedup_key] = accepted
            self._accepted_chat_messages.move_to_end(dedup_key)
            while len(self._accepted_chat_messages) > self._accepted_chat_messages_max:
                self._accepted_chat_messages.popitem(last=False)

        async def _chat_dedup_record_async(self, chat_id: str, client_message_id: str, accepted: dict[str, Any]) -> None:
            await asyncio.to_thread(self._chat_dedup_record, chat_id, client_message_id, accepted)

        def _chat_dedup_forget(self, chat_id: str, client_message_id: str) -> None:
            self._accepted_chat_messages.pop((chat_id, client_message_id), None)
            if hasattr(self._chat_event_logs, "dedup_forget"):
                try:
                    self._chat_event_logs.dedup_forget(chat_id, client_message_id)
                except Exception:  # noqa: BLE001
                    pass

        async def _chat_dedup_forget_async(self, chat_id: str, client_message_id: str) -> None:
            await asyncio.to_thread(self._chat_dedup_forget, chat_id, client_message_id)

        async def _cancel_active_turn(self, chat_id: str, active: dict[str, Any], *, reason: str) -> None:
            turn_id = str(active.get("turn_id") or "")
            if not turn_id or turn_id in self._cancelled_turns:
                return
            self._cancelled_turns.add(turn_id)
            active["cancelled"] = True
            await self._append_chat_event_async(chat_id, "turn.cancelled", {
                "turn_id": turn_id,
                "message_id": str(active.get("message_id") or ""),
                "reason": reason,
                "cancelled_at": _utc_iso(),
            })
            assistant_ids = self._assistant_messages_by_turn.get(turn_id) or set()
            if not assistant_ids:
                message_id = _message_id("asst")
                self._assistant_turn_by_message[message_id] = turn_id
                self._assistant_messages_by_turn.setdefault(turn_id, set()).add(message_id)
                assistant_ids = {message_id}
                await self._append_chat_event_async(chat_id, "message.created", {
                    "message_id": message_id,
                    "role": "hermes",
                    "text": "",
                    "status": "cancelled",
                    "created_at": _utc_iso(),
                    "turn_id": turn_id,
                })
            else:
                for message_id in assistant_ids:
                    await self._append_chat_event_async(chat_id, "message.updated", {
                        "message_id": message_id,
                        "turn_id": turn_id,
                        "status": "cancelled",
                        "updated_at": _utc_iso(),
                    })
            for activity_id in self._activity_ids_by_turn.pop(turn_id, set()):
                if self._activity_status_by_id.get(activity_id) in {"completed", "failed"}:
                    self._activity_payloads_by_id.pop(activity_id, None)
                    self._activity_status_by_id.pop(activity_id, None)
                    continue
                await self._append_activity_terminal(
                    chat_id,
                    turn_id,
                    activity_id,
                    status="failed",
                    error="Turn cancelled",
                )
                self._activity_payloads_by_id.pop(activity_id, None)
                self._activity_status_by_id.pop(activity_id, None)
            current = self._active_turns_by_chat.get(chat_id)
            if current and current.get("turn_id") == turn_id:
                self._active_turns_by_chat.pop(chat_id, None)

        def _append_cancel_marker_to_session_history(self, source: Any, turn_id: str) -> None:
            if not turn_id or turn_id in self._session_history_cancel_markers:
                return
            session_store = getattr(self, "_session_store", None)
            if session_store is None:
                return
            try:
                session_entry = session_store.get_or_create_session(source)
                session_store.append_to_transcript(session_entry.session_id, {
                    "role": "assistant",
                    "content": (
                        "[Hermes run cancelled by the user before completion. "
                        "Do not continue that cancelled request unless the user explicitly asks to resume it.]"
                    ),
                })
                self._session_history_cancel_markers.add(turn_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to append Vylen cancellation marker to Hermes session history: %s", exc)

        def _cleanup_cancelled_turn(self, chat_id: str, turn_id: str) -> None:
            assistant_message_ids = self._assistant_messages_by_turn.pop(turn_id, set())
            for assistant_message_id in assistant_message_ids:
                self._assistant_turn_by_message.pop(assistant_message_id, None)
            for activity_id in self._activity_ids_by_turn.pop(turn_id, set()):
                self._activity_payloads_by_id.pop(activity_id, None)
                self._activity_status_by_id.pop(activity_id, None)
            self._cancelled_turns.discard(turn_id)
            active = self._active_turns_by_chat.get(chat_id)
            if active and active.get("turn_id") == turn_id:
                self._active_turns_by_chat.pop(chat_id, None)

        async def _dispatch_native_stop(self, chat_id: str, active: dict[str, Any]) -> None:
            source = active.get("source")
            if source is None:
                return
            task = asyncio.current_task()
            if task is not None:
                self._suppress_response_tasks.add(task)
            try:
                await self.handle_message(self._build_command_event(
                    source=source,
                    text="/stop",
                    turn_id=str(active.get("turn_id") or ""),
                ))
            finally:
                if task is not None:
                    self._suppress_response_tasks.discard(task)

        async def _dispatch_native_command(
            self,
            frame: dict[str, Any],
            chat_id: str,
            text: str,
            *,
            suppress_confirm: bool,
            expected_confirm_commands: set[str] | None = None,
            wait_for_completion: bool = False,
        ) -> bool:
            source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
            if source is None:
                return False
            session_key = self._session_key_for_source(source)
            task = asyncio.current_task()
            if task is not None:
                self._suppress_response_tasks.add(task)
            if suppress_confirm and session_key:
                self._native_confirm_sessions[session_key] = {
                    "deadline": time.time() + self._action_ttl_seconds,
                    "commands": set(expected_confirm_commands or ()),
                }
            try:
                await self.handle_message(self._build_command_event(
                    source=source,
                    text=text,
                    turn_id="",
                ))
                if wait_for_completion and session_key:
                    await self._wait_for_native_command_completion(session_key)
            finally:
                if task is not None:
                    self._suppress_response_tasks.discard(task)
                if wait_for_completion and suppress_confirm and session_key:
                    self._native_confirm_sessions.pop(session_key, None)
            return True

        async def _wait_for_native_command_completion(self, session_key: str) -> None:
            task = getattr(self, "_session_tasks", {}).get(session_key)
            if task is None or task is asyncio.current_task():
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=min(self._action_ttl_seconds, 30.0))
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Timed out waiting for native command completion") from exc

        def _session_task_running(self, session_key: str) -> bool:
            if not session_key:
                return False
            task = getattr(self, "_session_tasks", {}).get(session_key)
            return bool(task is not None and task is not asyncio.current_task() and not task.done())

        def _source_for_chat(self, chat_id: str):
            active = self._active_turns_by_chat.get(chat_id)
            if active is not None and active.get("source") is not None:
                return active.get("source")
            for accepted in reversed(self._accepted_chat_messages.values()):
                source = accepted.get("source")
                if source is not None and str(getattr(source, "chat_id", "") or "") == chat_id:
                    return source
            return None

        def _source_for_chat_action(self, frame: dict[str, Any], chat_id: str, message_id: str | None = None):
            user_id = str(frame.get("user_id") or "")
            if not user_id:
                return None
            from gateway.session import SessionSource

            return SessionSource(
                platform=self.platform,
                chat_id=chat_id,
                chat_name=str(frame.get("chat_name") or chat_id),
                chat_type="dm",
                user_id=user_id,
                user_name=str(frame.get("user_name") or user_id),
                message_id=message_id or _message_id("command"),
            )

        def _session_key_for_source(self, source: Any) -> str:
            from gateway.session import build_session_key

            return build_session_key(
                source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            )

        async def _sync_hermes_session_title(self, frame: dict[str, Any], chat_id: str, title: str) -> None:
            source = self._source_for_chat_action(frame, chat_id) or self._source_for_chat(chat_id)
            if source is None:
                return

            def sync() -> None:
                session_store = getattr(self, "_session_store", None)
                if session_store is None:
                    return
                db = getattr(session_store, "_db", None) or getattr(self, "_session_db", None)
                if db is None:
                    return
                session_entry = session_store.get_or_create_session(source)
                clean_title = title
                sanitizer = getattr(db, "sanitize_title", None)
                if callable(sanitizer):
                    clean_title = sanitizer(title)
                if clean_title:
                    db.set_session_title(session_entry.session_id, clean_title)

            try:
                await asyncio.to_thread(sync)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to sync Vylen chat title to Hermes session: %s", type(exc).__name__)

        async def _sync_chat_title_from_hermes_later(self, chat_id: str, source: Any) -> None:
            await asyncio.sleep(0.5)

            def read_title() -> str:
                session_store = getattr(self, "_session_store", None)
                if session_store is None:
                    return ""
                db = getattr(session_store, "_db", None) or getattr(self, "_session_db", None)
                if db is None or not hasattr(db, "get_session_title"):
                    return ""
                session_entry = session_store.get_or_create_session(source)
                return str(db.get_session_title(session_entry.session_id) or "").strip()

            try:
                title = await asyncio.to_thread(read_title)
                if not title or not hasattr(self._chat_event_logs, "get_chat"):
                    return
                chat = await asyncio.to_thread(self._chat_event_logs.get_chat, chat_id)
                if chat is None or str(chat.title or "").strip() == title:
                    return
                allowed_titles = {"", "New conversation", _derive_chat_title_from_text(await self._first_user_message_text(chat_id))}
                if str(chat.title or "").strip() not in allowed_titles:
                    return
                if hasattr(self._chat_event_logs, "rename_chat"):
                    await asyncio.to_thread(self._chat_event_logs.rename_chat, chat_id, title)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to sync Hermes auto-title to Vylen chat: %s", exc)

        async def _first_user_message_text(self, chat_id: str) -> str:
            def read() -> str:
                if not hasattr(self._chat_event_logs, "replay_after"):
                    return ""
                for event in self._chat_event_logs.replay_after(chat_id, 0, limit=50):
                    if event.kind != "message.created":
                        continue
                    payload = event.payload if isinstance(event.payload, dict) else {}
                    if str(payload.get("role") or "") == "user":
                        return str(payload.get("text") or "")
                return ""

            try:
                return await asyncio.to_thread(read)
            except Exception:
                return ""

        async def _append_session_status(self, chat_id: str, *, source: Any | None = None) -> None:
            active = self._active_turns_by_chat.get(chat_id)
            source = source or self._source_for_chat(chat_id)
            session_key = self._session_key_for_source(source) if source is not None else ""
            active_turn_id = str((active or {}).get("turn_id") or "")
            running_activities = [
                activity_id
                for activity_id in self._activity_ids_by_turn.get(active_turn_id, set())
                if self._activity_status_by_id.get(activity_id) == "running"
            ] if active_turn_id else []
            queued, queued_exact = self._queue_depth_for_session(session_key)
            await self._append_chat_event_async(chat_id, "session.status", {
                "status_id": _message_id("status"),
                "state": "running" if active_turn_id else ("queued" if queued else "idle"),
                "running": bool(active_turn_id),
                "turn_id": active_turn_id,
                "running_activities": len(running_activities),
                "queued": queued,
                "queued_exact": queued_exact,
                "updated_at": _utc_iso(),
            })

        async def _append_session_controls(self, chat_id: str, *, source: Any | None = None) -> None:
            source = source or self._source_for_chat(chat_id)
            session_key = self._session_key_for_source(source) if source is not None else ""
            model, provider = self._current_model_provider(source, session_key)
            effort, scope, display = self._current_reasoning_controls(source, session_key)
            await self._append_chat_event_async(chat_id, "session.controls", {
                "controls_id": _message_id("controls"),
                "model": model,
                "provider": provider,
                "reasoning_effort": effort,
                "reasoning_scope": scope,
                "reasoning_display": display,
                "updated_at": _utc_iso(),
            })

        def _gateway_runner(self) -> Any | None:
            runner = getattr(self, "_runner", None)
            if runner is not None:
                return runner
            try:
                from gateway import run as gateway_run

                runner_ref = getattr(gateway_run, "_gateway_runner_ref", None)
                return runner_ref() if callable(runner_ref) else None
            except Exception:  # noqa: BLE001
                return None

        def _current_model_provider(self, source: Any | None, session_key: str) -> tuple[str, str]:
            runner = self._gateway_runner()
            resolver = getattr(runner, "_resolve_session_agent_runtime", None) if runner is not None else None
            if callable(resolver):
                try:
                    model, runtime = resolver(source=source, session_key=session_key)
                    provider = runtime.get("provider") if isinstance(runtime, dict) else ""
                    return str(model or "").strip(), str(provider or "").strip()
                except Exception:  # noqa: BLE001
                    logger.debug("vylen gateway: failed to resolve runtime model config", exc_info=True)

            model = ""
            provider = ""
            try:
                from gateway.run import _load_gateway_config

                cfg = _load_gateway_config() or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    model = str(model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name") or "").strip()
                    provider = str(model_cfg.get("provider") or "").strip()
                elif model_cfg:
                    model = str(model_cfg).strip()
            except Exception:  # noqa: BLE001
                logger.debug("vylen gateway: failed to read model config", exc_info=True)

            override_owner = runner if runner is not None else self
            override = (getattr(override_owner, "_session_model_overrides", {}) or {}).get(session_key, {}) if session_key else {}
            if isinstance(override, dict):
                model = str(override.get("model") or model).strip()
                provider = str(override.get("provider") or provider).strip()
            return model, provider

        def _current_reasoning_controls(self, source: Any | None, session_key: str) -> tuple[str, str, bool]:
            runner = self._gateway_runner()
            resolver_owner = runner if runner is not None else self
            try:
                resolver = getattr(resolver_owner, "_resolve_session_reasoning_config")
                config = resolver(source=source, session_key=session_key)
            except Exception:  # noqa: BLE001
                logger.debug("vylen gateway: failed to resolve reasoning config", exc_info=True)
                config = None

            if isinstance(config, dict) and config.get("enabled") is False:
                effort = "none"
            elif isinstance(config, dict) and str(config.get("effort") or "").strip().lower() in {"minimal", "low", "medium", "high", "xhigh"}:
                effort = str(config.get("effort") or "").strip().lower()
            else:
                effort = "medium"

            overrides = getattr(resolver_owner, "_session_reasoning_overrides", {}) or {}
            scope = "session" if session_key and session_key in overrides else "global"
            display = self._current_reasoning_display(source)
            return effort, scope, display

        def _current_reasoning_display(self, source: Any | None) -> bool:
            try:
                from gateway.run import _load_gateway_config, _platform_config_key

                cfg = _load_gateway_config() or {}
                display_cfg = cfg.get("display") if isinstance(cfg, dict) else {}
                if isinstance(display_cfg, dict):
                    platform = getattr(source, "platform", "") if source is not None else ""
                    platform_key = _platform_config_key(platform) if platform else ""
                    platforms_cfg = display_cfg.get("platforms")
                    if platform_key and isinstance(platforms_cfg, dict):
                        platform_cfg = platforms_cfg.get(platform_key)
                        if isinstance(platform_cfg, dict) and "show_reasoning" in platform_cfg:
                            return bool(platform_cfg.get("show_reasoning"))
                    if "show_reasoning" in display_cfg:
                        return bool(display_cfg.get("show_reasoning"))
            except Exception:  # noqa: BLE001
                logger.debug("vylen gateway: failed to read reasoning display config", exc_info=True)

            try:
                return bool(self._load_show_reasoning())
            except Exception:  # noqa: BLE001
                return False

        def _should_queue_new_message(self, chat_id: str, event: Any) -> bool:
            active = self._active_turns_by_chat.get(chat_id)
            if active is not None:
                return True
            try:
                session_key = self._session_key_for_source(event.source)
            except Exception:  # noqa: BLE001
                return False
            if session_key in getattr(self, "_active_sessions", {}):
                return True
            task = getattr(self, "_session_tasks", {}).get(session_key)
            return bool(task is not None and not task.done())

        def _enqueue_message_event(self, event: Any) -> None:
            session_key = self._session_key_for_source(event.source)
            runner = getattr(self, "_runner", None)
            if runner is None:
                try:
                    from gateway import run as gateway_run

                    runner_ref = getattr(gateway_run, "_gateway_runner_ref", None)
                    runner = runner_ref() if callable(runner_ref) else None
                except Exception:  # noqa: BLE001
                    runner = None
            enqueue_fifo = getattr(runner, "_enqueue_fifo", None) if runner is not None else None
            if callable(enqueue_fifo):
                enqueue_fifo(session_key, event, self)
                return
            queued_events = getattr(self, "_queued_events", None)
            if queued_events is None:
                queued_events = {}
                self._queued_events = queued_events
            if session_key in self._pending_messages:
                queued_events.setdefault(session_key, []).append(event)
            else:
                self._pending_messages[session_key] = event

        def _promote_local_queued_event(self, event: Any) -> None:
            try:
                session_key = self._session_key_for_source(event.source)
            except Exception:  # noqa: BLE001
                return
            if not session_key or session_key in getattr(self, "_pending_messages", {}):
                return
            queued_events = getattr(self, "_queued_events", None)
            if not isinstance(queued_events, dict):
                return
            overflow = queued_events.get(session_key)
            if not overflow:
                return
            next_event = overflow.pop(0)
            if not overflow:
                queued_events.pop(session_key, None)
            self._pending_messages[session_key] = next_event

        def _queue_depth_for_session(self, session_key: str) -> tuple[int, bool]:
            if not session_key:
                return 0, True
            runner = getattr(self, "_runner", None)
            if runner is None:
                try:
                    from gateway import run as gateway_run

                    runner_ref = getattr(gateway_run, "_gateway_runner_ref", None)
                    runner = runner_ref() if callable(runner_ref) else None
                except Exception:  # noqa: BLE001
                    runner = None
            queue_depth = getattr(runner, "_queue_depth", None) if runner is not None else None
            if callable(queue_depth):
                try:
                    return max(0, int(queue_depth(session_key, adapter=self))), True
                except Exception:  # noqa: BLE001
                    pass

            queued = 1 if session_key in getattr(self, "_pending_messages", {}) else 0
            queued_events = getattr(self, "_queued_events", None)
            if isinstance(queued_events, dict):
                queued += len(queued_events.get(session_key, []) or [])
            return queued, queued == 0

        def _consume_native_confirm_session(self, session_key: str, confirm_id: str) -> bool:
            marker = self._native_confirm_sessions.get(session_key)
            if marker is None:
                return False
            if isinstance(marker, dict):
                deadline = float(marker.get("deadline") or 0)
                expected_commands = set(marker.get("commands") or [])
            else:
                deadline = float(marker or 0)
                expected_commands = set()
            if deadline <= time.time():
                self._native_confirm_sessions.pop(session_key, None)
                return False
            try:
                from tools import slash_confirm as _slash_confirm_mod

                pending = _slash_confirm_mod.get_pending(session_key)
            except Exception:  # noqa: BLE001
                pending = None
            if not pending or str(pending.get("confirm_id") or "") != confirm_id:
                self._native_confirm_sessions.pop(session_key, None)
                return False
            pending_command = str(pending.get("command") or "")
            if expected_commands and pending_command not in expected_commands:
                self._native_confirm_sessions.pop(session_key, None)
                return False
            self._native_confirm_sessions.pop(session_key, None)
            return True

        def _active_turn_for_event(self, event) -> dict[str, Any]:
            from gateway.session import build_session_key

            return {
                "turn_id": _event_turn_id(event),
                "message_id": str(getattr(event, "message_id", "") or ""),
                "session_key": build_session_key(
                    event.source,
                    group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                    thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
                ),
                "source": event.source,
            }

        async def _emit_tool_progress(
            self,
            chat_id: str,
            turn_id: str,
            progress_message_id: str,
            progress: list[dict[str, str]],
        ) -> None:
            target_message_id = await self._ensure_activity_message(chat_id, turn_id)
            seen: set[str] = set()
            current_ids: list[str] = []
            for index, item in enumerate(progress):
                activity_id = _activity_id(turn_id, progress_message_id, index, item)
                if activity_id in seen:
                    continue
                seen.add(activity_id)
                current_ids.append(activity_id)
                signature = (item["tool"], item.get("label") or "", item.get("emoji") or "")
                existing_signature = self._activity_payloads_by_id.get(activity_id)
                if existing_signature == signature:
                    continue
                self._activity_payloads_by_id[activity_id] = signature
                self._activity_ids_by_turn.setdefault(turn_id, set()).add(activity_id)
                payload = {
                    "turn_id": turn_id,
                    "message_id": target_message_id,
                    "activity_id": activity_id,
                    "tool": item["tool"],
                    "label": item.get("label") or "",
                    "emoji": item.get("emoji") or "",
                    "status": "running",
                }
                if existing_signature is None:
                    payload["started_at"] = _utc_iso()
                    event_kind = "activity.started"
                else:
                    payload["updated_at"] = _utc_iso()
                    event_kind = "activity.updated"
                self._activity_status_by_id[activity_id] = "running"
                await self._append_chat_event_async(chat_id, event_kind, payload)
            for activity_id in current_ids[:-1]:
                if self._activity_status_by_id.get(activity_id) != "running":
                    continue
                await self._append_activity_terminal(
                    chat_id,
                    turn_id,
                    activity_id,
                    status="completed",
                    message_id=target_message_id,
                )

        async def _append_activity_terminal(
            self,
            chat_id: str,
            turn_id: str,
            activity_id: str,
            *,
            status: str,
            message_id: str | None = None,
            error: str | None = None,
        ) -> None:
            payload: dict[str, Any] = {
                "turn_id": turn_id,
                "activity_id": activity_id,
                "status": status,
                "updated_at": _utc_iso(),
            }
            if message_id:
                payload["message_id"] = message_id
            if error:
                payload["error"] = error
            self._activity_status_by_id[activity_id] = status
            await self._append_chat_event_async(chat_id, "activity.completed", payload)

        async def _ensure_activity_message(self, chat_id: str, turn_id: str) -> str:
            existing = self._activity_message_by_turn.get(turn_id)
            if existing:
                return existing
            existing_messages = self._assistant_messages_by_turn.get(turn_id) or set()
            if existing_messages:
                chosen = next(iter(existing_messages))
                self._activity_message_by_turn[turn_id] = chosen
                return chosen
            message_id = _message_id("asst")
            self._activity_message_by_turn[turn_id] = message_id
            self._assistant_turn_by_message[message_id] = turn_id
            self._assistant_messages_by_turn.setdefault(turn_id, set()).add(message_id)
            await self._append_chat_event_async(chat_id, "message.created", {
                "message_id": message_id,
                "role": "hermes",
                "text": "",
                "status": "running",
                "created_at": _utc_iso(),
                "turn_id": turn_id,
            })
            return message_id

        async def _emit_action_expired(
            self,
            chat_id: str,
            action_id: str,
            kind: str,
            record: dict[str, Any] | None,
            *,
            reason: str,
        ) -> None:
            if record is not None and record.get("expired_emitted"):
                return
            event_kind = "approval.expired" if kind == "approval" else "confirm.expired"
            payload = {
                "turn_id": (record or {}).get("turn_id"),
                "action_id": action_id,
                "message_id": (record or {}).get("message_id", ""),
                "reason": reason,
                "updated_at": _utc_iso(),
            }
            await self._append_chat_event_async(chat_id, event_kind, payload)
            if record is not None:
                record["expired_emitted"] = True
                self._action_cards.pop(action_id, None)

        async def _send_chat_message_error(self, frame: dict[str, Any], code: str, message: str) -> None:
            await self._send_frame({
                "type": FRAME_CHAT_MESSAGE_ERROR,
                "request_id": str(frame.get("request_id") or ""),
                "chat_id": str(frame.get("chat_id") or ""),
                "client_message_id": str(frame.get("client_message_id") or ""),
                "code": code,
                "message": message,
            })

        async def _send_chat_action_error(self, frame: dict[str, Any], code: str, message: str) -> None:
            await self._send_frame({
                "type": FRAME_CHAT_ACTION_ERROR,
                "request_id": str(frame.get("request_id") or ""),
                "chat_id": str(frame.get("chat_id") or ""),
                "action_id": str(frame.get("action_id") or ""),
                "turn_id": str(frame.get("turn_id") or ""),
                "code": code,
                "message": message,
            })

        async def _send_message_control_error(
            self,
            frame: dict[str, Any],
            ack_type: str,
            code: str,
            message: str,
        ) -> None:
            if ack_type == FRAME_CHAT_ACTION_ACK:
                await self._send_chat_action_error(frame, code, message)
                return
            await self._send_chat_message_error(frame, code, message)

        async def _send_frame(self, frame: dict[str, Any]) -> None:
            if self._client is None:
                logger.warning("vylen gateway: cannot send frame without socket: %s", frame.get("type"))
                return
            await self._client.send(frame)

    return VylenGatewayAdapter


def _parse_cron_envelope(text: str) -> tuple[str, str, str]:
    """Strip Hermes's cron-response chrome from `text` and extract job_id +
    job_name. Returns `(body, job_id, job_name)`. If the envelope doesn't
    match (chat message, ad-hoc send, or upstream changed the format),
    returns the raw text and empty strings for the IDs — preserving the
    pre-cron behavior so chat sends still work.
    """
    match = _CRON_ENVELOPE_RE.match(text)
    if not match:
        return text, "", ""
    body = match.group("body").strip()
    return body, match.group("job_id").strip(), match.group("name").strip()


def _native_chat_command(text: str) -> bool:
    command, args = _split_native_chat_command(text)
    if command in {"status", "reset"}:
        return not args
    return command in {"queue", "steer"}


def _split_native_chat_command(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", ""
    command, _sep, args = stripped[1:].partition(" ")
    command = command.lower()
    if command == "q":
        command = "queue"
    return command, args.strip()


def _push_cursor_chat_id(chat_id: Any) -> str:
    # Vylen currently renders every plugin-initiated push in one synthetic
    # notifications inbox, and the app subscribes to that single cursor stream.
    return VYLEN_INBOX_CHAT_ID


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    if not _ID_RE.match(text):
        return ""
    return text


def _message_id(prefix: str) -> str:
    return f"msg_{prefix}_{uuid.uuid4().hex}"


def _event_turn_id(event: Any) -> str:
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, dict):
        turn_id = raw.get("turn_id")
        if isinstance(turn_id, str):
            return turn_id
    return ""


def _utc_iso(epoch_seconds: float | None = None) -> str:
    ts = time.time() if epoch_seconds is None else epoch_seconds
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    header, sep, encoded = data_url.partition(",")
    if sep != "," or not header.startswith("data:"):
        raise ValueError("attachment data_url must be a data URL")
    metadata = header[5:]
    parts = metadata.split(";")
    mime_type = parts[0].lower()
    if not mime_type or "base64" not in parts[1:]:
        raise ValueError("attachment data_url must be base64 encoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("attachment data_url is not valid base64") from exc
    if not data:
        raise ValueError("attachment is empty")
    return mime_type, data


def _parse_tool_progress(content: Any) -> list[dict[str, str]]:
    if not isinstance(content, str):
        return []
    parsed: list[dict[str, str]] = []
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []
    for line in lines:
        clean = _DEDUP_SUFFIX_RE.sub("", line)
        match = _TOOL_PROGRESS_LINE_RE.match(clean)
        if not match:
            return []
        label = (match.group("label") or "").strip()
        if len(label) >= 2 and label[0] == label[-1] == '"':
            label = label[1:-1]
        parsed.append({
            "emoji": match.group("emoji") or "",
            "tool": match.group("tool"),
            "label": label,
        })
    return parsed


def _chat_state_error_code(exc: Exception, *, fallback: str = "CHAT_STATE_UNAVAILABLE") -> str:
    if isinstance(exc, EventTooLarge):
        return "CHAT_EVENT_TOO_LARGE"
    if isinstance(exc, ChatStateUnavailable):
        return exc.code
    if isinstance(exc, InvalidChatStateEvent):
        return exc.code
    return fallback


def _activity_id(
    turn_id: str,
    progress_message_id: str,
    index: int,
    item: dict[str, str],
) -> str:
    raw = f"{turn_id}\x00{progress_message_id}\x00{index}\x00{item.get('tool', '')}"
    return "act_" + uuid.uuid5(uuid.NAMESPACE_URL, raw).hex


def adapter_factory(config):
    """Hermes calls this with a PlatformConfig and expects an adapter instance."""
    cls = make_adapter_class()
    return cls(config)


def check_dependencies() -> bool:
    """Hermes calls this before instantiation to verify deps. We need:
    - VYLEN_INSTANCE_TOKEN set
    - websockets importable (it's our own dep so it always is when we are)
    """
    try:
        load_from_env()
    except ConfigError as exc:
        logger.info("Vylen gateway not configured: %s", exc)
        return False
    return True


def _authorize_vylen_user(user_id: str) -> None:
    """Authorize the cloud-authenticated Vylen owner for Hermes' platform gate."""
    clean = str(user_id or "").strip()
    if not clean:
        return
    existing = [
        part.strip()
        for part in os.environ.get(VYLEN_ALLOWED_USERS_ENV, "").split(",")
        if part.strip()
    ]
    if clean in existing:
        return
    os.environ[VYLEN_ALLOWED_USERS_ENV] = ",".join([*existing, clean])
