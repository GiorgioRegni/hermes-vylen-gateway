from __future__ import annotations

import asyncio
import json
import time

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


class FailingChunkWriter(CaptureWriter):
    def __init__(self, fail_after_chunks: int) -> None:
        super().__init__()
        self.fail_after_chunks = fail_after_chunks

    async def send_chunk(self, chunk: bytes) -> None:
        if len(self.chunks) >= self.fail_after_chunks:
            raise RuntimeError("client disconnected")
        await super().send_chunk(chunk)


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


class BlockingAPI(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def _run_agent(self, **kwargs):
        await self.release.wait()
        return await super()._run_agent(**kwargs)


class DeltaThenBlockAPI(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _run_agent(self, **kwargs):
        cb = kwargs.get("stream_delta_callback")
        if cb:
            cb("partial")
        self.started.set()
        await self.release.wait()
        return await super()._run_agent(**kwargs)


class CountingAPI(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def _run_agent(self, **kwargs):
        self.calls += 1
        result, usage = await super()._run_agent(**kwargs)
        final_response = f"hello: {kwargs.get('user_message')}"
        result["final_response"] = final_response
        result["messages"] = [{"role": "assistant", "content": final_response}]
        return result, usage


class BlockingCountingAPI(CountingAPI):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _run_agent(self, **kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        result, usage = await FakeAPI._run_agent(self, **kwargs)
        final_response = f"hello: {kwargs.get('user_message')}"
        result["final_response"] = final_response
        result["messages"] = [{"role": "assistant", "content": final_response}]
        return result, usage


class CancellableAPI(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _run_agent(self, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FailingOnceAPI(CountingAPI):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def _run_agent(self, **kwargs):
        self.calls += 1
        if not self.failed:
            self.failed = True
            raise RuntimeError("boom")
        return await FakeAPI._run_agent(self, **kwargs)


class FailedResultAPI(CountingAPI):
    async def _run_agent(self, **kwargs):
        self.calls += 1
        return {
            "failed": True,
            "final_response": None,
            "error": "provider auth failed",
            "messages": [],
            "session_id": kwargs.get("session_id") or "sid",
        }, {"input_tokens": 4, "output_tokens": 0, "total_tokens": 4}


class RaisingAPI(FakeAPI):
    async def _run_agent(self, **kwargs):
        cb = kwargs.get("stream_delta_callback")
        if cb:
            cb("partial")
        raise RuntimeError("provider crashed")


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


def _output_text(payload):
    return payload["output"][0]["content"][0]["text"]


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
async def test_responses_stream_store_false_string_does_not_persist():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "stream": True, "store": "false"})

    assert writer.status == 200
    assert b"event: response.completed" in writer.body
    assert api._response_store.responses == {}


@pytest.mark.asyncio
async def test_stream_response_failed_result_emits_failed_event():
    api = FailedResultAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "stream": True})

    assert writer.status == 200
    assert b"event: response.failed" in writer.body
    assert b"event: response.completed" not in writer.body
    stored = next(iter(api._response_store.responses.values()))
    assert stored["response"]["status"] == "failed"
    assert stored["response"]["error"] == {"message": "provider auth failed", "type": "server_error"}


@pytest.mark.asyncio
async def test_stream_chat_agent_exception_emits_error_not_stop():
    runner = InProcessAgentRunner(api_adapter=RaisingAPI())

    writer = await _dispatch(
        runner,
        "POST",
        "/v1/chat/completions",
        {"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert writer.status == 200
    assert b"event: error" in writer.body
    assert b"provider crashed" in writer.body
    assert b'"finish_reason": "stop"' not in writer.body
    assert writer.body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_chat_failed_result_emits_error_not_stop():
    runner = InProcessAgentRunner(api_adapter=FailedResultAPI())

    writer = await _dispatch(
        runner,
        "POST",
        "/v1/chat/completions",
        {"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert writer.status == 200
    assert b"event: error" in writer.body
    assert b"provider auth failed" in writer.body
    assert b'"finish_reason": "stop"' not in writer.body
    assert writer.body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_chat_completion_failed_result_returns_error():
    runner = InProcessAgentRunner(api_adapter=FailedResultAPI())

    writer = await _dispatch(
        runner,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )

    payload = json.loads(writer.body)
    assert writer.status == 500
    assert payload["error"] == {
        "message": "provider auth failed",
        "type": "server_error",
        "param": None,
        "code": None,
    }


@pytest.mark.asyncio
async def test_responses_idempotency_key_returns_cached_response():
    api = CountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)

    first_payload = json.loads(first.body)
    second_payload = json.loads(second.body)
    assert first.status == 200
    assert second.status == 200
    assert second_payload["id"] == first_payload["id"]
    assert api.calls == 1


@pytest.mark.asyncio
async def test_responses_store_false_string_does_not_persist():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)

    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "store": "false"})

    assert writer.status == 200
    assert json.loads(writer.body)["status"] == "completed"
    assert api._response_store.responses == {}


@pytest.mark.asyncio
async def test_responses_store_null_preserves_default_persistence():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)

    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "store": None})

    assert writer.status == 200
    assert api._response_store.responses


@pytest.mark.asyncio
async def test_responses_store_unknown_string_preserves_default_persistence():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)

    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "store": "bogus"})

    assert writer.status == 200
    assert api._response_store.responses


@pytest.mark.asyncio
async def test_responses_malformed_stream_flag_uses_non_streaming_default():
    api = FakeAPI()
    runner = InProcessAgentRunner(api_adapter=api)

    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "stream": {"value": True}})

    assert writer.status == 200
    assert writer.headers["Content-Type"] == "application/json"
    assert json.loads(writer.body)["status"] == "completed"


@pytest.mark.asyncio
async def test_responses_idempotency_key_does_not_cache_different_input():
    api = CountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/responses", {"input": "different"}, headers=headers)

    first_payload = json.loads(first.body)
    second_payload = json.loads(second.body)
    assert first.status == 200
    assert second.status == 200
    assert second_payload["id"] != first_payload["id"]
    assert _output_text(second_payload) == "hello: different"
    assert api.calls == 2


@pytest.mark.asyncio
async def test_responses_idempotency_key_does_not_cache_different_fingerprint_field():
    api = CountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "model": "a"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi", "model": "b"}, headers=headers)

    first_payload = json.loads(first.body)
    second_payload = json.loads(second.body)
    assert first.status == 200
    assert second.status == 200
    assert second_payload["id"] != first_payload["id"]
    assert api.calls == 2


@pytest.mark.asyncio
async def test_responses_idempotency_key_does_not_cache_different_conversation_history():
    api = CountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(
        runner,
        "POST",
        "/v1/responses",
        {"input": "hi", "conversation_history": [{"role": "user", "content": "first context"}]},
        headers=headers,
    )
    second = await _dispatch(
        runner,
        "POST",
        "/v1/responses",
        {"input": "hi", "conversation_history": [{"role": "user", "content": "second context"}]},
        headers=headers,
    )

    first_payload = json.loads(first.body)
    second_payload = json.loads(second.body)
    assert first.status == 200
    assert second.status == 200
    assert second_payload["id"] != first_payload["id"]
    assert api.calls == 2


@pytest.mark.asyncio
async def test_responses_idempotency_same_key_different_inflight_fingerprints_run_separately():
    api = BlockingCountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = asyncio.create_task(_dispatch(runner, "POST", "/v1/responses", {"input": "first"}, headers=headers))
    second = asyncio.create_task(_dispatch(runner, "POST", "/v1/responses", {"input": "second"}, headers=headers))
    for _ in range(50):
        if api.calls == 2:
            break
        await asyncio.sleep(0.01)
    assert api.calls == 2

    api.release.set()
    first_writer, second_writer = await asyncio.gather(first, second)

    first_payload = json.loads(first_writer.body)
    second_payload = json.loads(second_writer.body)
    assert _output_text(first_payload) == "hello: first"
    assert _output_text(second_payload) == "hello: second"


@pytest.mark.asyncio
async def test_responses_idempotency_retry_survives_cancelled_first_request():
    api = BlockingCountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}
    writer = CaptureWriter()
    first = asyncio.create_task(
        runner.dispatch(
            "POST",
            "/v1/responses",
            headers,
            json.dumps({"input": "hi"}).encode("utf-8"),
            writer,
        )
    )
    await asyncio.wait_for(api.started.wait(), timeout=1.0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    api.release.set()
    retry = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)

    assert retry.status == 200
    assert json.loads(retry.body)["status"] == "completed"
    assert api.calls == 1


@pytest.mark.asyncio
async def test_responses_idempotency_cache_expires():
    api = CountingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)
    first_payload = json.loads(first.body)
    runner._sweep_orphaned_runs_once(now=time.time() + 4_000.0)
    second = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)

    second_payload = json.loads(second.body)
    assert second_payload["id"] != first_payload["id"]
    assert api.calls == 2


@pytest.mark.asyncio
async def test_responses_idempotency_key_failure_is_not_cached():
    api = FailingOnceAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    with pytest.raises(RuntimeError):
        await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)
    retry = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)

    assert retry.status == 200
    assert api.calls == 2


@pytest.mark.asyncio
async def test_responses_failed_result_returns_and_stores_failed_response():
    api = FailedResultAPI()
    runner = InProcessAgentRunner(api_adapter=api)

    writer = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"})

    payload = json.loads(writer.body)
    assert writer.status == 200
    assert payload["status"] == "failed"
    assert payload["error"] == {"message": "provider auth failed", "type": "server_error"}
    assert _output_text(payload) == "provider auth failed"
    stored = api._response_store.get(payload["id"])
    assert stored["response"] == payload
    assert stored["conversation_history"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_responses_failed_result_is_idempotently_replayed():
    api = FailedResultAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/responses", {"input": "hi"}, headers=headers)

    first_payload = json.loads(first.body)
    second_payload = json.loads(second.body)
    assert second_payload == first_payload
    assert second_payload["status"] == "failed"
    assert api.calls == 1


@pytest.mark.asyncio
async def test_responses_rejects_invalid_idempotency_key():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    writer = await _dispatch(
        runner,
        "POST",
        "/v1/responses",
        {"input": "hi"},
        headers={"Content-Type": "application/json", "Idempotency-Key": "bad\nkey"},
    )

    assert writer.status == 400
    assert json.loads(writer.body)["error"]["message"] == "Invalid Idempotency-Key"


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


@pytest.mark.asyncio
async def test_run_events_can_be_replayed_after_first_consumer_finishes():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})
    run_id = json.loads(created.body)["run_id"]

    first = await _dispatch(runner, "GET", f"/v1/runs/{run_id}/events")
    second = await _dispatch(runner, "GET", f"/v1/runs/{run_id}/events")

    assert first.status == 200
    assert second.status == 200
    assert b"run.completed" in first.body
    assert b"run.completed" in second.body


@pytest.mark.asyncio
async def test_run_events_after_cursor_replays_only_newer_events():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})
    run_id = json.loads(created.body)["run_id"]

    replay = await _dispatch(runner, "GET", f"/v1/runs/{run_id}/events?after=1")

    assert replay.status == 200
    assert b"message.delta" not in replay.body
    assert b"run.completed" in replay.body


@pytest.mark.asyncio
async def test_run_event_log_survives_sse_client_disconnect_while_active():
    api = DeltaThenBlockAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})
    run_id = json.loads(created.body)["run_id"]
    await asyncio.wait_for(api.started.wait(), timeout=1.0)

    dropped_writer = FailingChunkWriter(fail_after_chunks=0)
    with pytest.raises(RuntimeError, match="client disconnected"):
        await runner.dispatch(
            "GET",
            f"/v1/runs/{run_id}/events",
            {"Content-Type": "application/json"},
            b"",
            dropped_writer,
        )

    assert runner._run_event_logs.get(run_id) is not None
    api.release.set()
    replay = await _dispatch(runner, "GET", f"/v1/runs/{run_id}/events")
    assert replay.status == 200
    assert b"message.delta" in replay.body
    assert b"run.completed" in replay.body


@pytest.mark.asyncio
async def test_runs_reject_missing_previous_response_id():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())

    writer = await _dispatch(
        runner,
        "POST",
        "/v1/runs",
        {"input": "hi", "previous_response_id": "resp_missing"},
    )

    assert writer.status == 404
    assert json.loads(writer.body)["error"]["message"] == "Previous response not found: resp_missing"


@pytest.mark.asyncio
async def test_runs_idempotency_key_returns_cached_run_for_same_body():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"}, headers=headers)

    assert first.status == 202
    assert second.status == 202
    assert json.loads(second.body)["run_id"] == json.loads(first.body)["run_id"]


@pytest.mark.asyncio
async def test_runs_idempotency_key_different_body_allocates_new_run():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())
    headers = {"Content-Type": "application/json", "Idempotency-Key": "retry-key"}

    first = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"}, headers=headers)
    second = await _dispatch(runner, "POST", "/v1/runs", {"input": "different"}, headers=headers)

    assert first.status == 202
    assert second.status == 202
    assert json.loads(second.body)["run_id"] != json.loads(first.body)["run_id"]


@pytest.mark.asyncio
async def test_completed_poll_only_runs_do_not_exhaust_active_run_limit():
    runner = InProcessAgentRunner(api_adapter=FakeAPI())

    for _ in range(10):
        created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})
        assert created.status == 202

    for _ in range(50):
        if runner._active_run_count() == 0:
            break
        await asyncio.sleep(0.01)
    assert runner._active_run_count() == 0
    assert len(runner._run_event_logs) == 10

    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})

    assert created.status == 202


@pytest.mark.asyncio
async def test_sweeper_does_not_drop_active_run_handles_by_age():
    api = BlockingAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    created = await _dispatch(runner, "POST", "/v1/runs", {"input": "hi"})
    run_id = json.loads(created.body)["run_id"]

    for _ in range(50):
        if runner._active_run_count() == 1:
            break
        await asyncio.sleep(0.01)
    assert runner._active_run_count() == 1
    runner._run_event_logs.get(run_id).created_at = 0.0

    runner._sweep_orphaned_runs_once(now=1_000.0)

    assert runner._run_event_logs.get(run_id) is not None
    assert run_id in runner._active_run_tasks
    assert run_id in runner._run_approval_sessions

    stopped = await _dispatch(runner, "POST", f"/v1/runs/{run_id}/stop", {})
    assert stopped.status == 200


@pytest.mark.asyncio
async def test_stream_response_cancels_spawned_agent_task():
    api = CancellableAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    writer = CaptureWriter()
    task = asyncio.create_task(
        runner.dispatch(
            "POST",
            "/v1/responses",
            {"Content-Type": "application/json"},
            json.dumps({"input": "hi", "stream": True}).encode("utf-8"),
            writer,
        )
    )
    await asyncio.wait_for(api.started.wait(), timeout=1.0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_chat_cancels_spawned_agent_task():
    api = CancellableAPI()
    runner = InProcessAgentRunner(api_adapter=api)
    writer = CaptureWriter()
    task = asyncio.create_task(
        runner.dispatch(
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            json.dumps({
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode("utf-8"),
            writer,
        )
    )
    await asyncio.wait_for(api.started.wait(), timeout=1.0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.cancelled.is_set()
