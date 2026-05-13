"""End-to-end test for the HTTP-tunnel relay path.

Sets up two in-process mocks: a fake Hermes HTTP server and a fake Vylen
Cloud WebSocket. Drives a `request` frame through the relay, verifies the
HTTP call hits the mock Hermes, and that the response is delivered back as
`response_headers` + `response_chunk` + `response_end` frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket

import httpx
import pytest
from aiohttp import web

from hermes_vylen_gateway.relay import (
    FRAME_REQUEST,
    FRAME_RESPONSE_CHUNK,
    FRAME_RESPONSE_END,
    FRAME_RESPONSE_HEADERS,
    HermesRelay,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_mock_hermes(port: int) -> web.AppRunner:
    """Mock Hermes that returns an SSE stream on POST /v1/responses."""

    async def handle_responses(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        # Echo back the input as the response.created text for the test.
        assert b'"input":"hi"' in body
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await response.write(
            b'event: response.created\ndata: {"response":{"id":"resp_1"}}\n\n'
        )
        await response.write(
            b'event: response.output_text.delta\ndata: {"delta":"hello"}\n\n'
        )
        await response.write(
            b'event: response.completed\ndata: {"response":{"id":"resp_1","status":"completed"}}\n\n'
        )
        return response

    app = web.Application()
    app.router.add_post("/v1/responses", handle_responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


@pytest.mark.asyncio
async def test_relay_streams_sse_from_hermes_back_to_cloud():
    hermes_port = _free_port()
    runner = await _start_mock_hermes(hermes_port)
    try:
        sent_frames: list[dict] = []
        sent_event = asyncio.Event()

        async def send(frame):
            sent_frames.append(frame)
            if frame["type"] == FRAME_RESPONSE_END:
                sent_event.set()

        relay = HermesRelay(send, hermes_url=f"http://127.0.0.1:{hermes_port}")
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
    finally:
        await runner.cleanup()

    # Frame ordering: headers → ≥1 chunk → end.
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
async def test_relay_emits_response_error_when_hermes_unreachable():
    sent_frames: list[dict] = []
    sent_event = asyncio.Event()

    async def send(frame):
        sent_frames.append(frame)
        if frame["type"] in {"response_error", FRAME_RESPONSE_END}:
            sent_event.set()

    # Pick an unbound port; httpx will fail to connect.
    bad_port = _free_port()
    relay = HermesRelay(send, hermes_url=f"http://127.0.0.1:{bad_port}")
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
    assert err["code"] == "HERMES_UNREACHABLE"
