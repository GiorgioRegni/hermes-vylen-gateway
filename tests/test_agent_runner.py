from __future__ import annotations

import json

import pytest

from hermes_vylen_gateway.agent_runner import InProcessAgentRunner


class CaptureWriter:
    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.finished = False

    async def send_headers(self, status: int, headers: dict[str, str]) -> None:
        self.status = status
        self.headers = headers

    async def send_chunk(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    async def finish(self) -> None:
        self.finished = True

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)


class Store:
    def __init__(self) -> None:
        self.responses: dict[str, dict] = {}
        self.conversations: dict[str, str] = {}

    def get(self, response_id: str):
        return self.responses.get(response_id)

    def put(self, response_id: str, data: dict) -> None:
        self.responses[response_id] = data

    def delete(self, response_id: str) -> bool:
        return self.responses.pop(response_id, None) is not None

    def get_conversation(self, name: str):
        return self.conversations.get(name)

    def set_conversation(self, name: str, response_id: str) -> None:
        self.conversations[name] = response_id


class FakeAPI:
    def __init__(self) -> None:
        self._model_name = "fake-hermes"
        self._response_store = Store()

    async def _run_agent(self, **kwargs):
        cb = kwargs.get("stream_delta_callback")
        if cb:
            cb("hello")
        return {
            "final_response": "hello",
            "messages": [{"role": "assistant", "content": "hello"}],
            "session_id": kwargs.get("session_id") or "sid",
        }, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}

    @staticmethod
    def _response_messages_turn_start_index(conversation_history, user_message, result):
        return 0

    @staticmethod
    def _extract_output_items(result, start_index=0):
        return [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": result["final_response"]}],
        }]

    @staticmethod
    def _build_response_conversation_history(conversation_history, user_message, result, final_response):
        return list(conversation_history) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_response},
        ]


async def _dispatch(runner, method, path, body=None, headers=None):
    writer = CaptureWriter()
    await runner.dispatch(
        method,
        path,
        headers or {"Content-Type": "application/json"},
        json.dumps(body).encode("utf-8") if body is not None else b"",
        writer,
    )
    return writer


@pytest.mark.asyncio
async def test_health_does_not_require_hermes_imports():
    runner = InProcessAgentRunner()
    writer = await _dispatch(runner, "GET", "/health")

    assert writer.status == 200
    assert json.loads(writer.body)["status"] == "ok"
    assert writer.finished is True


@pytest.mark.asyncio
async def test_unknown_path_returns_openai_404():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    writer = await _dispatch(runner, "GET", "/missing")

    assert writer.status == 404
    payload = json.loads(writer.body)
    assert payload["error"]["type"] == "not_found_error"


@pytest.mark.asyncio
async def test_invalid_json_returns_400():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    writer = CaptureWriter()
    await runner.dispatch("POST", "/v1/responses", {"Content-Type": "application/json"}, b"{", writer)

    assert writer.status == 400
    assert json.loads(writer.body)["error"]["message"] == "Invalid JSON in request body"


@pytest.mark.asyncio
async def test_responses_stream_persists_and_emits_sse():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "stream": True})

    assert writer.status == 200
    assert writer.headers["Content-Type"] == "text/event-stream"
    assert b"event: response.created" in writer.body
    assert b"event: response.output_text.delta" in writer.body
    assert b"event: response.completed" in writer.body
    assert api._response_store.responses


@pytest.mark.asyncio
async def test_response_get_and_delete():
    api = FakeAPI()
    api._response_store.put("resp_x", {"response": {"id": "resp_x", "status": "completed"}})
    runner = InProcessAgentRunner(api_adapter=api)

    got = await _dispatch(runner, "GET", "/v1/responses/resp_x")
    deleted = await _dispatch(runner, "DELETE", "/v1/responses/resp_x")

    assert got.status == 200
    assert json.loads(got.body)["id"] == "resp_x"
    assert deleted.status == 200
    assert json.loads(deleted.body)["deleted"] is True


@pytest.mark.asyncio
async def test_run_create_status_and_events():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})

    assert created.status == 202
    run_id = json.loads(created.body)["run_id"]

    events = await _dispatch(runner, "GET", f"/v1/runs/{run_id}/events")
    status = await _dispatch(runner, "GET", f"/v1/runs/{run_id}")

    assert events.status == 200
    assert b"run.completed" in events.body
    assert status.status == 200
    assert json.loads(status.body)["status"] == "completed"
