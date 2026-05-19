"""In-process OpenAI-compatible Hermes request dispatcher.

The cloud still sends HTTP-shaped request frames over the gateway WebSocket.
This module handles those frames without forwarding them to Hermes's loopback
``api_server``. At runtime it reuses Hermes's ``APIServerAdapter`` helpers and
state stores, but it writes bytes directly through the gateway response writer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import parse_qs, urlsplit

from .event_log import EventLogRegistry, ResumeExpired, RetainedEventLog

logger = logging.getLogger(__name__)

_MAX_REQUEST_BYTES = 10_000_000
_CHAT_KEEPALIVE_SECONDS = 30.0
_RUN_STREAM_TTL = 300
_RUN_STATUS_TTL = 3600
_MAX_CONCURRENT_RUNS = 10
_RUN_STREAM_MAX_EVENTS = 1000
_RUN_STREAM_MAX_BYTES = 4 * 1024 * 1024
_MAX_SESSION_HEADER_LEN = 256
_MAX_IDEMPOTENCY_KEY_LEN = 256
_IDEMPOTENCY_TTL = _RUN_STATUS_TTL


class StreamWriter(Protocol):
    async def send_headers(self, status: int, headers: dict[str, str]) -> None: ...
    async def send_chunk(self, chunk: bytes) -> None: ...
    async def finish(self) -> None: ...


@dataclass
class _ParsedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def header(self, name: str, default: str = "") -> str:
        lname = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lname:
                return value
        return default

    def json_body(self) -> dict[str, Any]:
        if len(self.body) > _MAX_REQUEST_BYTES:
            raise _RequestError(413, "Request body too large.", code="body_too_large")
        try:
            parsed = json.loads(self.body.decode("utf-8") if self.body else "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _RequestError(400, "Invalid JSON in request body") from None
        if not isinstance(parsed, dict):
            raise _RequestError(400, "JSON request body must be an object")
        return parsed


@dataclass
class _CachedResponse:
    fingerprint: str
    response: tuple[dict[str, Any], dict[str, str]]
    created_at: float


@dataclass
class _CachedRun:
    fingerprint: str
    run_id: str
    created_at: float


class _RequestError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        err_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param


class HermesUnavailable(Exception):
    pass


async def dispatch(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes | None,
    writer: StreamWriter,
) -> int:
    """Dispatch one HTTP-shaped gateway request in-process."""
    runner = _get_runner()
    return await runner.dispatch(method, path, headers, body or b"", writer)


def check_available() -> tuple[bool, str | None]:
    """Return whether Hermes internals needed for agent routes can import."""
    try:
        _get_runner().ensure_api()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


_RUNNER: "InProcessAgentRunner | None" = None


def _get_runner() -> "InProcessAgentRunner":
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = InProcessAgentRunner()
    return _RUNNER


class InProcessAgentRunner:
    """Stateful mirror of Hermes's API server request surface."""

    def __init__(self, api_adapter: Any | None = None, api_module: Any | None = None):
        self._api = api_adapter
        self._api_module = api_module
        self._init_error: Exception | None = None
        self._run_event_logs = EventLogRegistry(
            ttl_seconds=_RUN_STREAM_TTL,
            max_events=_RUN_STREAM_MAX_EVENTS,
            max_bytes=_RUN_STREAM_MAX_BYTES,
        )
        self._active_run_agents: dict[str, Any] = {}
        self._active_run_tasks: dict[str, asyncio.Task[Any]] = {}
        self._run_statuses: dict[str, dict[str, Any]] = {}
        self._run_approval_sessions: dict[str, str] = {}
        # Idempotency-Key → run cache. Fingerprinting prevents a reused key
        # from replaying an unrelated prompt/session.
        self._idempotency_keys: dict[str, _CachedRun] = {}
        self._response_idempotency_keys: dict[str, _CachedResponse] = {}
        self._response_idempotency_inflight: dict[
            tuple[str, str],
            asyncio.Task[tuple[dict[str, Any], dict[str, str]]],
        ] = {}
        self._sweep_task: asyncio.Task[Any] | None = None

    def ensure_api(self) -> Any:
        if self._api is not None:
            return self._api
        if self._init_error is not None:
            raise HermesUnavailable(str(self._init_error))
        try:
            from gateway.config import PlatformConfig
            from gateway.platforms import api_server

            self._api_module = api_server
            self._api = api_server.APIServerAdapter(
                PlatformConfig(enabled=True, extra={})
            )
        except Exception as exc:  # noqa: BLE001
            self._init_error = exc
            raise HermesUnavailable(str(exc)) from exc
        return self._api

    async def dispatch(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        writer: StreamWriter,
    ) -> int:
        req = _ParsedRequest(method=method.upper(), path=path, headers=headers, body=body)
        try:
            status = await self._dispatch(req, writer)
        except _RequestError as exc:
            status = await _write_json(
                writer,
                exc.status,
                _openai_error(exc.message, err_type=exc.err_type, code=exc.code, param=exc.param),
            )
        except HermesUnavailable as exc:
            status = await _write_json(
                writer,
                503,
                _openai_error(f"Hermes internals unavailable: {exc}", err_type="server_error"),
            )
        await writer.finish()
        return status

    async def _dispatch(self, req: _ParsedRequest, writer: StreamWriter) -> int:
        path = urlsplit(req.path).path
        if req.method == "GET" and path in {"/health", "/v1/health"}:
            return await _write_json(writer, 200, {"status": "ok", "platform": "hermes-agent"})
        if req.method == "GET" and path == "/health/detailed":
            return await _write_json(writer, 200, self._health_detailed())

        api = self.ensure_api()

        if req.method == "GET" and path == "/v1/models":
            return await self._models(writer, api)
        if req.method == "GET" and path == "/v1/capabilities":
            return await self._capabilities(writer, api)
        if req.method == "POST" and path == "/v1/chat/completions":
            return await self._chat_completions(req, writer, api)
        if req.method == "POST" and path == "/v1/responses":
            return await self._responses(req, writer, api)
        if path.startswith("/v1/responses/") and len(path.split("/")) == 4:
            response_id = path.rsplit("/", 1)[1]
            if req.method == "GET":
                return await self._get_response(writer, api, response_id)
            if req.method == "DELETE":
                return await self._delete_response(writer, api, response_id)
        if req.method == "POST" and path == "/v1/runs":
            return await self._create_run(req, writer, api)
        if path.startswith("/v1/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                run_id = parts[2]
                suffix = parts[3] if len(parts) == 4 else ""
                if req.method == "GET" and suffix == "":
                    return await self._get_run(writer, run_id)
                if req.method == "GET" and suffix == "events":
                    return await self._run_events(req, writer, run_id)
                if req.method == "POST" and suffix == "stop":
                    return await self._stop_run(writer, run_id)
                if req.method == "POST" and suffix == "approval":
                    return await self._run_approval(req, writer, run_id)

        return await _write_json(
            writer,
            404,
            _openai_error(f"Path not found: {req.method} {path}", err_type="not_found_error"),
        )

    def _health_detailed(self) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        try:
            from gateway.status import read_runtime_status

            runtime = read_runtime_status() or {}
        except Exception:
            runtime = {}
        return {
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": runtime.get("gateway_state"),
            "platforms": runtime.get("platforms", {}),
            "active_agents": runtime.get("active_agents", 0),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        }

    async def _models(self, writer: StreamWriter, api: Any) -> int:
        model = getattr(api, "_model_name", "hermes-agent")
        return await _write_json(
            writer,
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "hermes",
                        "permission": [],
                        "root": model,
                        "parent": None,
                    }
                ],
            },
        )

    async def _capabilities(self, writer: StreamWriter, api: Any) -> int:
        model = getattr(api, "_model_name", "hermes-agent")
        return await _write_json(
            writer,
            200,
            {
                "object": "hermes.api_server.capabilities",
                "platform": "hermes-agent",
                "model": model,
                "auth": {"type": "bearer", "required": False},
                "runtime": {
                    "mode": "server_agent",
                    "tool_execution": "server",
                    "split_runtime": False,
                    "description": "Vylen invokes Hermes in-process; tools execute on the Hermes host.",
                },
                "features": {
                    "chat_completions": True,
                    "chat_completions_streaming": True,
                    "responses_api": True,
                    "responses_streaming": True,
                    "run_submission": True,
                    "run_status": True,
                    "run_events_sse": True,
                    "run_stop": True,
                    "run_approval_response": True,
                    "tool_progress_events": True,
                    "approval_events": True,
                    "session_continuity_header": "X-Hermes-Session-Id",
                    "session_key_header": "X-Hermes-Session-Key",
                    "cors": False,
                },
                "endpoints": {
                    "health": {"method": "GET", "path": "/health"},
                    "health_detailed": {"method": "GET", "path": "/health/detailed"},
                    "models": {"method": "GET", "path": "/v1/models"},
                    "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                    "responses": {"method": "POST", "path": "/v1/responses"},
                    "runs": {"method": "POST", "path": "/v1/runs"},
                    "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                    "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                    "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                    "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
                },
            },
        )

    async def _chat_completions(self, req: _ParsedRequest, writer: StreamWriter, api: Any) -> int:
        body = req.json_body()
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise _RequestError(400, "Missing or invalid 'messages' field")

        system_prompt = None
        conversation_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                content = self._normalize_chat_content(raw_content)
                system_prompt = content if system_prompt is None else f"{system_prompt}\n{content}"
            elif role in {"user", "assistant"}:
                try:
                    content = self._normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    raise self._validation_error(exc, f"messages[{idx}].content") from None
                conversation_messages.append({"role": role, "content": content})

        user_message = conversation_messages[-1].get("content", "") if conversation_messages else ""
        history = conversation_messages[:-1]
        if not self._content_has_visible_payload(user_message):
            raise _RequestError(400, "No user message found in messages")

        gateway_session_key = _parse_session_key(req)
        session_id = req.header("X-Hermes-Session-Id").strip()
        if not session_id:
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = str(cm.get("content", ""))
                    break
            session_id = self._derive_chat_session_id(system_prompt, first_user)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model = str(body.get("model") or getattr(api, "_model_name", "hermes-agent"))
        created = int(time.time())
        if _coerce_request_bool(body.get("stream"), default=False):
            return await self._stream_chat(
                writer,
                api,
                completion_id,
                model,
                created,
                user_message,
                history,
                system_prompt,
                session_id,
                gateway_session_key,
            )

        result, usage = await api._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=gateway_session_key,
        )
        if isinstance(result, dict) and result.get("failed"):
            return await _write_json(
                writer,
                500,
                _openai_error(
                    str(result.get("error") or "agent run failed"),
                    err_type="server_error",
                ),
            )
        final = result.get("final_response") or ""
        response_headers = {"X-Hermes-Session-Id": result.get("session_id", session_id)}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        return await _write_json(
            writer,
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _chat_usage(usage),
            },
            response_headers,
        )

    async def _stream_chat(
        self,
        writer: StreamWriter,
        api: Any,
        completion_id: str,
        model: str,
        created: int,
        user_message: Any,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        session_id: str,
        gateway_session_key: str | None,
    ) -> int:
        headers = _sse_headers(session_id=session_id, gateway_session_key=gateway_session_key)
        await writer.send_headers(200, headers)
        loop = asyncio.get_running_loop()
        stream_q: asyncio.Queue[Any] = asyncio.Queue()
        started_tool_ids: set[str] = set()

        def on_delta(delta: Any) -> None:
            if delta is not None:
                loop.call_soon_threadsafe(stream_q.put_nowait, delta)

        def on_tool_start(tool_call_id: str, function_name: str, function_args: Any) -> None:
            if not tool_call_id or str(function_name).startswith("_"):
                return
            started_tool_ids.add(tool_call_id)
            loop.call_soon_threadsafe(stream_q.put_nowait, ("__tool_progress__", {
                "tool": function_name,
                "label": function_name,
                "toolCallId": tool_call_id,
                "status": "running",
            }))

        def on_tool_complete(tool_call_id: str, function_name: str, function_args: Any, function_result: Any) -> None:
            if not tool_call_id or tool_call_id not in started_tool_ids:
                return
            started_tool_ids.discard(tool_call_id)
            loop.call_soon_threadsafe(stream_q.put_nowait, ("__tool_progress__", {
                "tool": function_name,
                "toolCallId": tool_call_id,
                "status": "completed",
            }))

        agent_ref: list[Any] = [None]
        task = asyncio.ensure_future(api._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            stream_delta_callback=on_delta,
            tool_start_callback=on_tool_start,
            tool_complete_callback=on_tool_complete,
            agent_ref=agent_ref,
            gateway_session_key=gateway_session_key,
        ))
        task.add_done_callback(lambda _fut: stream_q.put_nowait(None))

        try:
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await _write_sse_data(writer, role_chunk)

            last_activity = time.monotonic()
            while True:
                try:
                    item = await asyncio.wait_for(stream_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    if time.monotonic() - last_activity >= _CHAT_KEEPALIVE_SECONDS:
                        await writer.send_chunk(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue
                if item is None:
                    break
                if isinstance(item, tuple) and item[0] == "__tool_progress__":
                    await _write_sse_event(writer, "hermes.tool.progress", item[1])
                else:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await _write_sse_data(writer, chunk)
                last_activity = time.monotonic()

            usage: dict[str, int] = {}
            stream_error: str | None = None
            try:
                result, usage = await task
                if isinstance(result, dict) and result.get("failed"):
                    stream_error = str(result.get("error") or "agent run failed")
            except Exception as exc:  # noqa: BLE001
                stream_error = str(exc)
                logger.warning("chat completion stream failed: %s", exc)
            if stream_error:
                await _write_sse_event(
                    writer,
                    "error",
                    _openai_error(stream_error, err_type="server_error"),
                )
                await writer.send_chunk(b"data: [DONE]\n\n")
                return 200
            finish = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": _chat_usage(usage),
            }
            await _write_sse_data(writer, finish)
            await writer.send_chunk(b"data: [DONE]\n\n")
            return 200
        except asyncio.CancelledError:
            await _cancel_stream_task(task, agent_ref, "SSE stream cancelled")
            raise

    async def _responses(self, req: _ParsedRequest, writer: StreamWriter, api: Any) -> int:
        body = req.json_body()
        raw_input = body.get("input")
        if raw_input is None:
            raise _RequestError(400, "Missing 'input' field")
        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = _coerce_request_bool(body.get("store"), default=True)
        if conversation and previous_response_id:
            raise _RequestError(400, "Cannot use both 'conversation' and 'previous_response_id'")
        if conversation:
            previous_response_id = api._response_store.get_conversation(conversation)

        input_messages = self._input_messages(raw_input)
        conversation_history: list[dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                raise _RequestError(400, "'conversation_history' must be an array of message objects")
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    raise _RequestError(400, f"conversation_history[{i}] must have 'role' and 'content' fields")
                try:
                    content = self._normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    raise self._validation_error(exc, f"conversation_history[{i}].content") from None
                conversation_history.append({"role": str(entry["role"]), "content": content})

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = api._response_store.get(previous_response_id)
            if stored is None:
                raise _RequestError(404, f"Previous response not found: {previous_response_id}")
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            if instructions is None:
                instructions = stored.get("instructions")

        conversation_history.extend(input_messages[:-1])
        user_message = input_messages[-1].get("content", "") if input_messages else ""
        if not self._content_has_visible_payload(user_message):
            raise _RequestError(400, "No user message found in input")
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        session_id = stored_session_id or str(uuid.uuid4())
        gateway_session_key = _parse_session_key(req)
        model = str(body.get("model") or getattr(api, "_model_name", "hermes-agent"))
        if _coerce_request_bool(body.get("stream"), default=False):
            return await self._stream_response(
                writer,
                api,
                model,
                conversation_history,
                user_message,
                instructions,
                conversation,
                bool(store),
                session_id,
                gateway_session_key,
            )

        idempotency_key = _parse_idempotency_key(req)
        response_headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key

        async def build_response() -> tuple[dict[str, Any], dict[str, str]]:
            result, usage = await api._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
            )
            final = result.get("final_response", "") or result.get("error", "(No response generated)")
            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            created_at = int(time.time())
            failed = bool(isinstance(result, dict) and result.get("failed"))
            if failed:
                output_items = [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": str(final)}],
                }]
            else:
                output_start = api._response_messages_turn_start_index(conversation_history, user_message, result)
                output_items = api._extract_output_items(result, start_index=output_start)
            response_data = {
                "id": response_id,
                "object": "response",
                "status": "failed" if failed else "completed",
                "created_at": created_at,
                "model": model,
                "output": output_items,
                "usage": _response_usage(usage),
            }
            if failed:
                response_data["error"] = {"message": str(final), "type": "server_error"}
            if store:
                if failed:
                    full_history = list(conversation_history)
                    full_history.append({"role": "user", "content": user_message})
                else:
                    full_history = api._build_response_conversation_history(
                        conversation_history, user_message, result, final
                    )
                api._response_store.put(response_id, {
                    "response": response_data,
                    "conversation_history": full_history,
                    "instructions": instructions,
                    "session_id": session_id,
                })
                if conversation:
                    api._response_store.set_conversation(conversation, response_id)
            return response_data, response_headers

        if idempotency_key is not None:
            fingerprint = _make_request_fingerprint(body)
            cached = self._response_idempotency_keys.get(idempotency_key)
            if cached is not None:
                if (
                    time.time() - cached.created_at <= _IDEMPOTENCY_TTL
                    and cached.fingerprint == fingerprint
                ):
                    cached_response, cached_headers = cached.response
                    return await _write_json(writer, 200, cached_response, cached_headers)
                if time.time() - cached.created_at > _IDEMPOTENCY_TTL:
                    self._response_idempotency_keys.pop(idempotency_key, None)

            inflight_key = (idempotency_key, fingerprint)
            response_task = self._response_idempotency_inflight.get(inflight_key)
            if response_task is None:
                async def build_and_cache() -> tuple[dict[str, Any], dict[str, str]]:
                    response = await build_response()
                    self._response_idempotency_keys[idempotency_key] = _CachedResponse(
                        fingerprint=fingerprint,
                        response=response,
                        created_at=time.time(),
                    )
                    return response

                response_task = asyncio.create_task(build_and_cache())
                self._response_idempotency_inflight[inflight_key] = response_task

                def clear_inflight(done_task: asyncio.Task[tuple[dict[str, Any], dict[str, str]]]) -> None:
                    if self._response_idempotency_inflight.get(inflight_key) is done_task:
                        self._response_idempotency_inflight.pop(inflight_key, None)

                response_task.add_done_callback(clear_inflight)
        else:
            response_data, response_headers = await build_response()
            return await _write_json(writer, 200, response_data, response_headers)
        try:
            response_data, response_headers = await asyncio.shield(response_task)
        except Exception:
            raise
        return await _write_json(writer, 200, response_data, response_headers)

    async def _stream_response(
        self,
        writer: StreamWriter,
        api: Any,
        model: str,
        conversation_history: list[dict[str, Any]],
        user_message: Any,
        instructions: str | None,
        conversation: str | None,
        store: bool,
        session_id: str,
        gateway_session_key: str | None,
    ) -> int:
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())
        await writer.send_headers(200, _sse_headers(session_id=session_id, gateway_session_key=gateway_session_key))
        loop = asyncio.get_running_loop()
        stream_q: asyncio.Queue[Any] = asyncio.Queue()

        def on_delta(delta: Any) -> None:
            if delta is not None:
                loop.call_soon_threadsafe(stream_q.put_nowait, delta)

        def on_tool_start(tool_call_id: str, function_name: str, function_args: Any) -> None:
            loop.call_soon_threadsafe(stream_q.put_nowait, ("__tool_started__", {
                "tool_call_id": tool_call_id,
                "name": function_name,
                "arguments": function_args or {},
            }))

        def on_tool_complete(tool_call_id: str, function_name: str, function_args: Any, function_result: Any) -> None:
            loop.call_soon_threadsafe(stream_q.put_nowait, ("__tool_completed__", {
                "tool_call_id": tool_call_id,
                "name": function_name,
                "arguments": function_args or {},
                "result": function_result,
            }))

        agent_ref: list[Any] = [None]
        task = asyncio.ensure_future(api._run_agent(
            user_message=user_message,
            conversation_history=conversation_history,
            ephemeral_system_prompt=instructions,
            session_id=session_id,
            stream_delta_callback=on_delta,
            tool_start_callback=on_tool_start,
            tool_complete_callback=on_tool_complete,
            agent_ref=agent_ref,
            gateway_session_key=gateway_session_key,
        ))
        task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        sequence = 0
        final_parts: list[str] = []
        emitted_items: list[dict[str, Any]] = []
        pending_tools: dict[str, dict[str, Any]] = {}
        output_index = 0
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_open = False
        message_output_index = 0

        async def event(name: str, payload: dict[str, Any]) -> None:
            nonlocal sequence
            payload.setdefault("sequence_number", sequence)
            sequence += 1
            await _write_sse_event(writer, name, payload)

        def envelope(status: str) -> dict[str, Any]:
            return {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }

        def persist(response_env: dict[str, Any], history: list[dict[str, Any]] | None = None) -> None:
            if not store:
                return
            if history is None:
                history = list(conversation_history)
                history.append({"role": "user", "content": user_message})
            api._response_store.put(response_id, {
                "response": response_env,
                "conversation_history": history,
                "instructions": instructions,
                "session_id": session_id,
            })
            if conversation:
                api._response_store.set_conversation(conversation, response_id)

        created_env = envelope("in_progress")
        created_env["output"] = []
        persist(created_env)
        await event("response.created", {"type": "response.created", "response": created_env})

        try:
            async def open_message() -> None:
                nonlocal message_open, message_output_index, output_index
                if message_open:
                    return
                message_open = True
                message_output_index = output_index
                output_index += 1
                await event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                })

            async def emit_delta(delta: str) -> None:
                await open_message()
                final_parts.append(delta)
                await event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta,
                    "logprobs": [],
                })

            async def emit_tool_started(payload: dict[str, Any]) -> None:
                nonlocal output_index
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{len(pending_tools) + 1}"
                args = payload.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": payload.get("name", ""),
                    "call_id": call_id,
                    "arguments": args_str,
                }
                idx = output_index
                output_index += 1
                pending_tools[call_id] = {"item": item, "index": idx}
                emitted_items.append({
                    "type": "function_call",
                    "name": item["name"],
                    "arguments": args_str,
                    "call_id": call_id,
                })
                await event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })

            async def emit_tool_completed(payload: dict[str, Any]) -> None:
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                pending = pending_tools.pop(call_id, None)
                if pending is None:
                    return
                item = dict(pending["item"])
                item["status"] = "completed"
                await event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["index"],
                    "item": item,
                })
                result = payload.get("result", "")
                result_str = result if isinstance(result, str) else json.dumps(result)
                out_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": result_str}],
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": out_item["output"],
                })
                await event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": out_item,
                })
                await event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": out_item,
                })

            last_activity = time.monotonic()
            while True:
                try:
                    item = await asyncio.wait_for(stream_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    if time.monotonic() - last_activity >= _CHAT_KEEPALIVE_SECONDS:
                        await writer.send_chunk(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue
                if item is None:
                    break
                if isinstance(item, tuple) and item[0] == "__tool_started__":
                    await emit_tool_started(item[1])
                elif isinstance(item, tuple) and item[0] == "__tool_completed__":
                    await emit_tool_completed(item[1])
                elif isinstance(item, str):
                    await emit_delta(item)
                last_activity = time.monotonic()

            usage: dict[str, int] = {}
            result: dict[str, Any] = {}
            agent_error: str | None = None
            try:
                result, usage = await task
                if result.get("failed"):
                    agent_error = str(result.get("error") or "agent run failed")
                elif result.get("final_response") and not final_parts:
                    await emit_delta(str(result["final_response"]))
            except Exception as exc:  # noqa: BLE001
                agent_error = str(exc)
                logger.exception("response stream failed")

            final_text = "".join(final_parts)
            if message_open:
                await event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_text,
                    "logprobs": [],
                })
                await event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": final_text}],
                    },
                })

            final_items = list(emitted_items)
            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text or (agent_error or "")}],
            })
            if agent_error:
                failed_env = envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": agent_error, "type": "server_error"}
                failed_env["usage"] = _response_usage(usage)
                persist(failed_env)
                await event("response.failed", {"type": "response.failed", "response": failed_env})
            else:
                completed_env = envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = _response_usage(usage)
                full_history = api._build_response_conversation_history(
                    conversation_history, user_message, result, final_text
                )
                persist(completed_env, full_history)
                await event("response.completed", {"type": "response.completed", "response": completed_env})
            return 200
        except asyncio.CancelledError:
            await _cancel_stream_task(task, agent_ref, "SSE stream cancelled")
            raise

    def _input_messages(self, raw_input: Any) -> list[dict[str, Any]]:
        if isinstance(raw_input, str):
            return [{"role": "user", "content": raw_input}]
        if not isinstance(raw_input, list):
            raise _RequestError(400, "'input' must be a string or array")
        messages: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_input):
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                try:
                    content = self._normalize_multimodal_content(item.get("content", ""))
                except ValueError as exc:
                    raise self._validation_error(exc, f"input[{idx}].content") from None
                messages.append({"role": item.get("role", "user"), "content": content})
        return messages

    async def _get_response(self, writer: StreamWriter, api: Any, response_id: str) -> int:
        stored = api._response_store.get(response_id)
        if stored is None:
            return await _write_json(writer, 404, _openai_error(f"Response not found: {response_id}"))
        return await _write_json(writer, 200, stored["response"])

    async def _delete_response(self, writer: StreamWriter, api: Any, response_id: str) -> int:
        if not api._response_store.delete(response_id):
            return await _write_json(writer, 404, _openai_error(f"Response not found: {response_id}"))
        return await _write_json(writer, 200, {"id": response_id, "object": "response", "deleted": True})

    async def _create_run(self, req: _ParsedRequest, writer: StreamWriter, api: Any) -> int:
        idempotency_key = _parse_idempotency_key(req)
        body = req.json_body()
        if idempotency_key is not None:
            fingerprint = _make_request_fingerprint(body)
            cached = self._idempotency_keys.get(idempotency_key)
            if cached is not None:
                if (
                    time.time() - cached.created_at <= _IDEMPOTENCY_TTL
                    and cached.fingerprint == fingerprint
                    and cached.run_id in self._run_statuses
                ):
                    return await _write_json(
                        writer,
                        202,
                        {"run_id": cached.run_id, "status": "started"},
                    )
                # Stale or evicted; drop and fall through to a fresh allocation.
                if time.time() - cached.created_at > _IDEMPOTENCY_TTL:
                    self._idempotency_keys.pop(idempotency_key, None)
        if self._active_run_count() >= _MAX_CONCURRENT_RUNS:
            return await _write_json(
                writer,
                429,
                _openai_error(f"Too many concurrent runs (max {_MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
            )
        raw_input = body.get("input")
        if not raw_input:
            raise _RequestError(400, "Missing 'input' field")
        user_message = raw_input if isinstance(raw_input, str) else (
            raw_input[-1].get("content", "") if isinstance(raw_input, list) and raw_input and isinstance(raw_input[-1], dict) else ""
        )
        if not user_message:
            raise _RequestError(400, "No user message found in input")
        conversation_history: list[dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if isinstance(raw_history, list):
            for entry in raw_history:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    conversation_history.append({"role": str(entry["role"]), "content": entry["content"]})
        previous_response_id = body.get("previous_response_id")
        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = api._response_store.get(previous_response_id)
            if stored is None:
                raise _RequestError(404, f"Previous response not found: {previous_response_id}")
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
        run_id = f"run_{uuid.uuid4().hex}"
        if idempotency_key is not None:
            self._idempotency_keys[idempotency_key] = _CachedRun(
                fingerprint=_make_request_fingerprint(body),
                run_id=run_id,
                created_at=time.time(),
            )
        session_id = body.get("session_id") or stored_session_id or run_id
        gateway_session_key = _parse_session_key(req)
        approval_session_key = gateway_session_key or session_id or run_id
        created_at = time.time()
        event_log = self._run_event_logs.get_or_create(run_id)
        self._run_approval_sessions[run_id] = approval_session_key
        self._set_run_status(
            run_id,
            "queued",
            created_at=created_at,
            session_id=session_id,
            model=body.get("model", getattr(api, "_model_name", "hermes-agent")),
        )
        self._ensure_sweeper()

        loop = asyncio.get_running_loop()

        def append_run_event(kind: str, payload: dict[str, Any]) -> None:
            try:
                if asyncio.get_running_loop() is loop:
                    event_log.append(kind, payload)
                    return
            except RuntimeError:
                pass
            loop.call_soon_threadsafe(event_log.append, kind, payload)

        def text_cb(delta: Any) -> None:
            if delta is None:
                return
            append_run_event("message.delta", {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": delta,
            })

        def event_cb(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            event = {
                "event": event_type,
                "run_id": run_id,
                "timestamp": time.time(),
                "tool": tool_name,
                "preview": preview,
            }
            append_run_event(event_type, event)

        async def run_and_close() -> None:
            try:
                self._set_run_status(run_id, "running")
                if hasattr(api, "_create_agent"):
                    result, usage = await self._run_agent_for_run(
                        api=api,
                        run_id=run_id,
                        event_log=event_log,
                        user_message=user_message,
                        conversation_history=conversation_history,
                        instructions=body.get("instructions"),
                        session_id=session_id,
                        approval_session_key=approval_session_key,
                        stream_delta_callback=text_cb,
                        tool_progress_callback=event_cb,
                        gateway_session_key=gateway_session_key,
                    )
                else:
                    result, usage = await api._run_agent(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        ephemeral_system_prompt=body.get("instructions"),
                        session_id=session_id,
                        stream_delta_callback=text_cb,
                        tool_progress_callback=event_cb,
                        gateway_session_key=gateway_session_key,
                    )
                if isinstance(result, dict) and result.get("failed"):
                    error_msg = result.get("error") or "agent run failed"
                    event_log.append("run.failed", {"event": "run.failed", "run_id": run_id, "timestamp": time.time(), "error": error_msg})
                    self._set_run_status(run_id, "failed", error=error_msg, last_event="run.failed")
                else:
                    output = result.get("final_response", "") if isinstance(result, dict) else ""
                    event_log.append("run.completed", {
                        "event": "run.completed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "output": output,
                        "usage": _response_usage(usage),
                    })
                    self._set_run_status(run_id, "completed", output=output, usage=_response_usage(usage), last_event="run.completed")
            except asyncio.CancelledError:
                self._set_run_status(run_id, "cancelled", last_event="run.cancelled")
                event_log.append("run.cancelled", {"event": "run.cancelled", "run_id": run_id, "timestamp": time.time()})
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("run %s failed", run_id)
                self._set_run_status(run_id, "failed", error=str(exc), last_event="run.failed")
                event_log.append("run.failed", {"event": "run.failed", "run_id": run_id, "timestamp": time.time(), "error": str(exc)})
            finally:
                event_log.close()
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)

        task = asyncio.create_task(run_and_close())
        self._active_run_tasks[run_id] = task
        response_headers = {"X-Hermes-Session-Key": gateway_session_key} if gateway_session_key else {}
        return await _write_json(writer, 202, {"run_id": run_id, "status": "started"}, response_headers)

    async def _run_agent_for_run(
        self,
        *,
        api: Any,
        run_id: str,
        event_log: RetainedEventLog,
        user_message: Any,
        conversation_history: list[dict[str, Any]],
        instructions: str | None,
        session_id: str,
        approval_session_key: str,
        stream_delta_callback: Any,
        tool_progress_callback: Any,
        gateway_session_key: str | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        loop = asyncio.get_running_loop()
        agent = api._create_agent(
            ephemeral_system_prompt=instructions,
            session_id=session_id,
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            gateway_session_key=gateway_session_key,
        )
        self._active_run_agents[run_id] = agent

        def approval_notify(approval_data: dict[str, Any]) -> None:
            event = dict(approval_data or {})
            event.update({
                "event": "approval.request",
                "run_id": run_id,
                "timestamp": time.time(),
                "choices": ["once", "session", "always", "deny"],
            })
            self._set_run_status(
                run_id,
                "waiting_for_approval",
                last_event="approval.request",
            )
            loop.call_soon_threadsafe(event_log.append, "approval.request", event)

        def run_sync() -> tuple[dict[str, Any], dict[str, int]]:
            approval_token = None
            session_tokens = []
            try:
                try:
                    from gateway.session_context import clear_session_vars, set_session_vars
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify,
                    )

                    approval_token = set_current_session_key(approval_session_key)
                    session_tokens = set_session_vars(
                        platform="api_server",
                        session_key=approval_session_key,
                    )
                    register_gateway_notify(approval_session_key, approval_notify)
                except Exception:
                    clear_session_vars = None
                    reset_current_session_key = None
                    unregister_gateway_notify = None

                result = agent.run_conversation(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    task_id=session_id or run_id,
                )
            finally:
                try:
                    if "unregister_gateway_notify" in locals() and unregister_gateway_notify is not None:
                        unregister_gateway_notify(approval_session_key)
                finally:
                    if approval_token is not None and "reset_current_session_key" in locals() and reset_current_session_key is not None:
                        try:
                            reset_current_session_key(approval_token)
                        except Exception:
                            pass
                    if session_tokens and "clear_session_vars" in locals() and clear_session_vars is not None:
                        try:
                            clear_session_vars(session_tokens)
                        except Exception:
                            pass
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            if isinstance(result, dict):
                effective_session_id = getattr(agent, "session_id", session_id)
                if isinstance(effective_session_id, str) and effective_session_id:
                    result["session_id"] = effective_session_id
            return result, usage

        return await loop.run_in_executor(None, run_sync)

    async def _get_run(self, writer: StreamWriter, run_id: str) -> int:
        status = self._run_statuses.get(run_id)
        if status is None:
            return await _write_json(writer, 404, _openai_error(f"Run not found: {run_id}", code="run_not_found"))
        return await _write_json(writer, 200, status)

    async def _run_events(self, req: _ParsedRequest, writer: StreamWriter, run_id: str) -> int:
        for _ in range(20):
            if self._run_event_logs.get(run_id) is not None:
                break
            await asyncio.sleep(0.05)
        else:
            return await _write_json(writer, 404, _openai_error(f"Run not found: {run_id}", code="run_not_found"))
        await writer.send_headers(200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        event_log = self._run_event_logs.get(run_id)
        if event_log is None:
            return 200
        after_seq = _parse_after_seq(req.path)
        try:
            async for retained in event_log.tail_after(after_seq, keepalive_seconds=30.0):
                event = dict(retained.payload) if isinstance(retained.payload, dict) else {"event": retained.kind, "data": retained.payload}
                event.setdefault("seq", retained.seq)
                await _write_sse_data(writer, event)
            await writer.send_chunk(b": stream closed\n\n")
        except ResumeExpired as exc:
            await _write_sse_data(writer, {
                "event": "run.resume_expired",
                "run_id": run_id,
                "code": "RUN_RESUME_EXPIRED",
                "floor_seq": exc.floor_seq,
                "latest_seq": exc.latest_seq,
            })
        return 200

    async def _stop_run(self, writer: StreamWriter, run_id: str) -> int:
        agent = self._active_run_agents.get(run_id)
        task = self._active_run_tasks.get(run_id)
        if agent is None and task is None:
            return await _write_json(writer, 404, _openai_error(f"Run not found: {run_id}", code="run_not_found"))
        self._set_run_status(run_id, "stopping", last_event="run.stopping")
        if agent is not None:
            try:
                agent.interrupt("Stop requested via API")
            except Exception:
                pass
        if task is not None and not task.done():
            task.cancel()
        return await _write_json(writer, 200, {"run_id": run_id, "status": "stopping"})

    async def _run_approval(self, req: _ParsedRequest, writer: StreamWriter, run_id: str) -> int:
        if run_id not in self._run_statuses:
            return await _write_json(writer, 404, _openai_error(f"Run not found: {run_id}", code="run_not_found"))
        body = req.json_body()
        choice = str(body.get("choice", "")).strip().lower()
        choice = {"approve": "once", "approved": "once", "allow": "once"}.get(choice, choice)
        if choice not in {"once", "session", "always", "deny"}:
            return await _write_json(
                writer,
                400,
                _openai_error("Invalid approval choice; expected one of: once, session, always, deny", code="invalid_approval_choice"),
            )
        approval_session_key = self._run_approval_sessions.get(run_id)
        if not approval_session_key:
            return await _write_json(writer, 409, _openai_error(f"Run has no active approval session: {run_id}", code="approval_not_active"))
        try:
            from tools.approval import resolve_gateway_approval

            resolved = resolve_gateway_approval(
                approval_session_key,
                choice,
                resolve_all=bool(body.get("all") or body.get("resolve_all")),
            )
        except Exception as exc:  # noqa: BLE001
            return await _write_json(writer, 500, _openai_error(str(exc)))
        if resolved <= 0:
            return await _write_json(writer, 409, _openai_error(f"Run has no pending approval: {run_id}", code="approval_not_pending"))
        self._set_run_status(run_id, "running", last_event="approval.responded")
        event_log = self._run_event_logs.get(run_id)
        if event_log is not None:
            event_log.append("approval.responded", {
                "event": "approval.responded",
                "run_id": run_id,
                "timestamp": time.time(),
                "choice": choice,
                "resolved": resolved,
            })
        return await _write_json(writer, 200, {
            "object": "hermes.run.approval_response",
            "run_id": run_id,
            "choice": choice,
            "resolved": resolved,
        })

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        current = self._run_statuses.get(run_id, {})
        current.update({
            "object": "hermes.run",
            "run_id": run_id,
            "status": status,
            "updated_at": now,
        })
        current.setdefault("created_at", fields.pop("created_at", now))
        current.update(fields)
        self._run_statuses[run_id] = current
        return current

    def _active_run_count(self) -> int:
        return sum(1 for task in self._active_run_tasks.values() if not task.done())

    def _ensure_sweeper(self) -> None:
        if self._sweep_task is None or self._sweep_task.done():
            self._sweep_task = asyncio.create_task(self._sweep_orphaned_runs())

    async def _sweep_orphaned_runs(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._sweep_orphaned_runs_once(time.time())

    def _sweep_orphaned_runs_once(self, now: float) -> None:
        for run_id, event_log in list(self._run_event_logs._logs.items()):
            if now - event_log.updated_at > _RUN_STREAM_TTL:
                task = self._active_run_tasks.get(run_id)
                if task is not None and not task.done():
                    continue
                self._run_event_logs.drop(run_id)
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)
        for run_id, status in list(self._run_statuses.items()):
            if status.get("status") in {"completed", "failed", "cancelled"} and now - float(status.get("updated_at", 0) or 0) > _RUN_STATUS_TTL:
                self._run_statuses.pop(run_id, None)
        for key, cached in list(self._idempotency_keys.items()):
            if now - cached.created_at > _IDEMPOTENCY_TTL:
                self._idempotency_keys.pop(key, None)
        for key, cached in list(self._response_idempotency_keys.items()):
            if now - cached.created_at > _IDEMPOTENCY_TTL:
                self._response_idempotency_keys.pop(key, None)

    def _normalize_chat_content(self, content: Any) -> str:
        fn = getattr(self._api_module, "_normalize_chat_content", None)
        if fn is not None:
            return fn(content)
        return _fallback_normalize_chat_content(content)

    def _normalize_multimodal_content(self, content: Any) -> Any:
        fn = getattr(self._api_module, "_normalize_multimodal_content", None)
        if fn is not None:
            return fn(content)
        return _fallback_normalize_chat_content(content)

    def _content_has_visible_payload(self, content: Any) -> bool:
        fn = getattr(self._api_module, "_content_has_visible_payload", None)
        if fn is not None:
            return fn(content)
        return bool(str(content).strip()) if not isinstance(content, list) else bool(content)

    @staticmethod
    def _derive_chat_session_id(system_prompt: str | None, first_user_message: str) -> str:
        import hashlib

        seed = f"{system_prompt or ''}\n{first_user_message}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"api-{digest}"

    @staticmethod
    def _validation_error(exc: ValueError, param: str) -> _RequestError:
        raw = str(exc)
        code, _, message = raw.partition(":")
        if not message:
            code, message = "invalid_content_part", raw
        return _RequestError(400, message, code=code, param=param)


def _parse_idempotency_key(req: _ParsedRequest) -> str | None:
    raw = req.header("Idempotency-Key", "").strip()
    if not raw:
        return None
    if re.search(r"[\r\n\x00]", raw):
        raise _RequestError(400, "Invalid Idempotency-Key")
    if len(raw) > _MAX_IDEMPOTENCY_KEY_LEN:
        raise _RequestError(400, "Idempotency-Key too long")
    return raw


def _make_request_fingerprint(body: dict[str, Any]) -> str:
    from hashlib import sha256

    fingerprint_body = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(fingerprint_body.encode("utf-8")).hexdigest()


def _coerce_request_bool(value: Any, *, default: bool = False) -> bool:
    # OpenAI-compatible clients sometimes send booleans as JSON strings
    # ("true"/"false") or numeric scalars. Unknown shapes fall back to the
    # caller's endpoint-specific default instead of using Python truthiness.
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return default
    return default


def _parse_session_key(req: _ParsedRequest) -> str | None:
    raw = req.header("X-Hermes-Session-Key", "").strip()
    if not raw:
        return None
    if re.search(r"[\r\n\x00]", raw):
        raise _RequestError(400, "Invalid session key")
    if len(raw) > _MAX_SESSION_HEADER_LEN:
        raise _RequestError(400, "Session key too long")
    return raw


def _parse_after_seq(path: str) -> int:
    query = parse_qs(urlsplit(path).query)
    for key in ("after_seq", "after", "cursor"):
        raw = query.get(key, [""])[0]
        if not raw:
            continue
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    return 0


async def _write_json(
    writer: StreamWriter,
    status: int,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> int:
    out_headers = {"Content-Type": "application/json"}
    if headers:
        out_headers.update(headers)
    await writer.send_headers(status, out_headers)
    await writer.send_chunk(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))
    return status


async def _write_sse_event(writer: StreamWriter, event: str, payload: dict[str, Any]) -> None:
    await writer.send_chunk(
        f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n".encode("utf-8")
    )


async def _write_sse_data(writer: StreamWriter, payload: Any) -> None:
    await writer.send_chunk(
        f"data: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n".encode("utf-8")
    )


async def _cancel_stream_task(
    task: asyncio.Future[Any],
    agent_ref: list[Any] | None,
    reason: str,
) -> None:
    agent = agent_ref[0] if agent_ref else None
    if agent is not None:
        try:
            agent.interrupt(reason)
        except Exception:
            pass
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("stream task ended during cancellation cleanup: %s", exc)


def _sse_headers(
    *,
    session_id: str | None = None,
    gateway_session_key: str | None = None,
) -> dict[str, str]:
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id
    if gateway_session_key:
        headers["X-Hermes-Session-Key"] = gateway_session_key
    return headers


def _openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def _chat_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0),
        "completion_tokens": int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
    }


def _response_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    return {
        "input_tokens": int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
    }


def _fallback_normalize_chat_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)
