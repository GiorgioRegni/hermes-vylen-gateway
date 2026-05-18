"""End-to-end tests for gateway request frames through the relay."""

from __future__ import annotations

import asyncio
import base64

import pytest

from hermes_vylen_gateway import agent_runner
from hermes_vylen_gateway.relay import (
    FRAME_REQUEST,
    FRAME_RESPONSE_CHUNK,
    FRAME_RESPONSE_END,
    FRAME_RESPONSE_HEADERS,
    HermesRelay,
)
from hermes_vylen_gateway.response_buffer import ResponseBufferRegistry


async def _write_sample_response(writer) -> int:
    await writer.send_headers(200, {"Content-Type": "text/event-stream"})
    await writer.send_chunk(
        b'event: response.created\ndata: {"response":{"id":"resp_1"}}\n\n'
    )
    await writer.send_chunk(
        b'event: response.output_text.delta\ndata: {"delta":"hello"}\n\n'
    )
    await writer.send_chunk(
        b'event: response.completed\ndata: {"response":{"id":"resp_1","status":"completed"}}\n\n'
    )
    await writer.finish()
    return 200


@pytest.mark.asyncio
async def test_relay_streams_in_process_response_back_to_cloud(monkeypatch):
    async def fake_dispatch(method, path, headers, body, writer):
        assert method == "POST"
        assert path == "/v1/responses"
        assert body == b'{"input":"hi","stream":true}'
        return await _write_sample_response(writer)

    monkeypatch.setattr(agent_runner, "dispatch", fake_dispatch)

    sent_frames: list[dict] = []
    sent_event = asyncio.Event()

    async def send(frame):
        sent_frames.append(frame)
        if frame["type"] == FRAME_RESPONSE_END:
            sent_event.set()

    relay = HermesRelay(send)
    try:
        req = {
            "type": FRAME_REQUEST,
            "request_id": "req_test",
            "method": "POST",
            "path": "/v1/responses",
            "headers": {"Content-Type": "application/json"},
            "body": base64.b64encode(b'{"input":"hi","stream":true}').decode(),
            "stream": True,
        }
        await relay.handle(req)
        await asyncio.wait_for(sent_event.wait(), timeout=3.0)
    finally:
        await relay.close()

    assert sent_frames[0]["type"] == FRAME_RESPONSE_HEADERS
    assert sent_frames[0]["status"] == 200
    headers_lower = {k.lower(): v for k, v in sent_frames[0]["headers"].items()}
    assert headers_lower.get("content-type") == "text/event-stream"
    assert sent_frames[-1]["type"] == FRAME_RESPONSE_END
    assert all(f["request_id"] == "req_test" for f in sent_frames)

    chunks = [f for f in sent_frames if f["type"] == FRAME_RESPONSE_CHUNK]
    assert chunks, "no response_chunk frames emitted"
    body = b"".join(base64.b64decode(c["data"]) for c in chunks)
    assert b"event: response.created" in body
    assert b"event: response.output_text.delta" in body
    assert b"event: response.completed" in body


@pytest.mark.asyncio
async def test_relay_emits_response_error_when_dispatch_crashes(monkeypatch):
    async def fake_dispatch(method, path, headers, body, writer):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_runner, "dispatch", fake_dispatch)
    sent_frames: list[dict] = []
    sent_event = asyncio.Event()

    async def send(frame):
        sent_frames.append(frame)
        if frame["type"] in {"response_error", FRAME_RESPONSE_END}:
            sent_event.set()

    relay = HermesRelay(send)
    try:
        await relay.handle({
            "type": FRAME_REQUEST,
            "request_id": "req_x",
            "method": "GET",
            "path": "/health",
            "headers": {},
            "body": "",
        })
        await asyncio.wait_for(sent_event.wait(), timeout=3.0)
    finally:
        await relay.close()

    err = sent_frames[-1]
    assert err["type"] == "response_error"
    assert err["request_id"] == "req_x"
    assert err["code"] == "RELAY_ERROR"


@pytest.mark.asyncio
async def test_relay_populates_response_buffer_for_resume(monkeypatch):
    async def fake_dispatch(method, path, headers, body, writer):
        return await _write_sample_response(writer)

    monkeypatch.setattr(agent_runner, "dispatch", fake_dispatch)
    sent_event = asyncio.Event()

    async def send(frame):
        if frame["type"] == FRAME_RESPONSE_END:
            sent_event.set()

    buffers = ResponseBufferRegistry(grace_seconds=300.0, max_bytes=1 << 20)
    relay = HermesRelay(send, response_buffers=buffers)
    try:
        await relay.handle({
            "type": FRAME_REQUEST,
            "request_id": "req_buf",
            "method": "POST",
            "path": "/v1/responses",
            "headers": {"Content-Type": "application/json"},
            "body": base64.b64encode(b'{"input":"hi","stream":true}').decode(),
            "stream": True,
        })
        await asyncio.wait_for(sent_event.wait(), timeout=3.0)
    finally:
        await relay.close()

    buf = buffers.get("resp_1")
    assert buf is not None, "relay should have created a buffer keyed by response_id"
    assert buf.complete is True
    assert buf.status == 200
    body = b"".join(buf.chunks)
    assert b"event: response.created" in body
    assert b"event: response.output_text.delta" in body
    assert b"event: response.completed" in body
