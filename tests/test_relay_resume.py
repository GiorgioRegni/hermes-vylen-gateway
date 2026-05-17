"""Tests for the response_resume frame: live-tail and replay-from-cursor."""

from __future__ import annotations

import asyncio
import base64
import socket
from typing import Any

import pytest
from aiohttp import web

from hermes_vylen_gateway.relay import (
    FRAME_REQUEST,
    FRAME_RESPONSE_CHUNK,
    FRAME_RESPONSE_END,
    FRAME_RESPONSE_ERROR,
    FRAME_RESPONSE_HEADERS,
    FRAME_RESPONSE_RESUME,
    HermesRelay,
)
from hermes_vylen_gateway.response_buffer import ResponseBufferRegistry


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_paced_hermes(port: int, gate: asyncio.Event) -> web.AppRunner:
    """Emits response.created immediately, one delta, then waits on `gate`
    before emitting the next delta + completed. Lets the test interleave a
    resume call between deltas."""

    async def handle_responses(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await response.write(
            b'event: response.created\ndata: {"response":{"id":"resp_live"}}\n\n'
        )
        await response.write(
            b'event: response.output_text.delta\ndata: {"delta":"first"}\n\n'
        )
        await gate.wait()
        await response.write(
            b'event: response.output_text.delta\ndata: {"delta":"second"}\n\n'
        )
        await response.write(
            b'event: response.completed\ndata: {"response":{"id":"resp_live","status":"completed"}}\n\n'
        )
        return response

    app = web.Application()
    app.router.add_post("/v1/responses", handle_responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


def _collect_for(request_id: str, frames: list[dict[str, Any]]) -> bytes:
    body = b""
    for f in frames:
        if f.get("request_id") == request_id and f["type"] == FRAME_RESPONSE_CHUNK:
            body += base64.b64decode(f["data"])
    return body


@pytest.mark.asyncio
async def test_resume_replays_from_cursor_after_completion():
    hermes_port = _free_port()
    gate = asyncio.Event()
    runner = await _start_paced_hermes(hermes_port, gate)
    try:
        frames: list[dict[str, Any]] = []
        original_done = asyncio.Event()
        resume_done = asyncio.Event()

        async def send(frame):
            frames.append(frame)
            t = frame["type"]
            rid = frame.get("request_id")
            if t == FRAME_RESPONSE_END and rid == "req_orig":
                original_done.set()
            elif t == FRAME_RESPONSE_END and rid == "req_resume":
                resume_done.set()

        buffers = ResponseBufferRegistry(grace_seconds=300.0, max_bytes=1 << 20)
        relay = HermesRelay(
            send,
            hermes_url=f"http://127.0.0.1:{hermes_port}",
            response_buffers=buffers,
        )
        try:
            await relay.handle({
                "type": FRAME_REQUEST,
                "request_id": "req_orig",
                "method": "POST",
                "path": "/v1/responses",
                "headers": {"Content-Type": "application/json"},
                "body": base64.b64encode(b'{"input":"hi","stream":true}').decode(),
                "stream": True,
            })
            gate.set()  # let mock Hermes finish
            await asyncio.wait_for(original_done.wait(), timeout=3.0)

            # Now resume from cursor 0 — should replay everything.
            await relay.handle_resume({
                "type": FRAME_RESPONSE_RESUME,
                "request_id": "req_resume",
                "response_id": "resp_live",
                "after_cursor": 0,
            })
            await asyncio.wait_for(resume_done.wait(), timeout=3.0)
        finally:
            await relay.close()
    finally:
        await runner.cleanup()

    original_body = _collect_for("req_orig", frames)
    resume_body = _collect_for("req_resume", frames)
    assert resume_body == original_body
    assert b"response.created" in resume_body
    assert b"response.completed" in resume_body

    # Resume past-cursor: should return only the bytes after that cursor.
    resume_frames = [
        f for f in frames if f.get("request_id") == "req_resume" and f["type"] == FRAME_RESPONSE_CHUNK
    ]
    assert resume_frames, "resume should emit at least one chunk"
    # Headers frame also present.
    resume_headers = next(
        f for f in frames if f.get("request_id") == "req_resume" and f["type"] == FRAME_RESPONSE_HEADERS
    )
    assert resume_headers["status"] == 200


@pytest.mark.asyncio
async def test_resume_returns_unknown_for_missing_response_id():
    sent_frames: list[dict[str, Any]] = []
    done = asyncio.Event()

    async def send(frame):
        sent_frames.append(frame)
        if frame["type"] == FRAME_RESPONSE_ERROR:
            done.set()

    buffers = ResponseBufferRegistry()
    relay = HermesRelay(send, response_buffers=buffers)
    try:
        await relay.handle_resume({
            "type": FRAME_RESPONSE_RESUME,
            "request_id": "req_miss",
            "response_id": "resp_does_not_exist",
            "after_cursor": 0,
        })
        await asyncio.wait_for(done.wait(), timeout=2.0)
    finally:
        await relay.close()

    err = sent_frames[-1]
    assert err["type"] == FRAME_RESPONSE_ERROR
    assert err["code"] == "RESUME_UNKNOWN"
    assert err["request_id"] == "req_miss"


@pytest.mark.asyncio
async def test_resume_tails_in_flight_response():
    hermes_port = _free_port()
    gate = asyncio.Event()
    runner = await _start_paced_hermes(hermes_port, gate)
    try:
        frames: list[dict[str, Any]] = []
        original_first_chunk = asyncio.Event()
        original_done = asyncio.Event()
        resume_done = asyncio.Event()

        async def send(frame):
            frames.append(frame)
            t = frame["type"]
            rid = frame.get("request_id")
            if t == FRAME_RESPONSE_CHUNK and rid == "req_orig" and not original_first_chunk.is_set():
                # Mark once we've seen the first chunk on the original tunnel.
                original_first_chunk.set()
            if t == FRAME_RESPONSE_END and rid == "req_orig":
                original_done.set()
            if t == FRAME_RESPONSE_END and rid == "req_resume":
                resume_done.set()

        buffers = ResponseBufferRegistry(grace_seconds=300.0, max_bytes=1 << 20)
        relay = HermesRelay(
            send,
            hermes_url=f"http://127.0.0.1:{hermes_port}",
            response_buffers=buffers,
        )
        try:
            await relay.handle({
                "type": FRAME_REQUEST,
                "request_id": "req_orig",
                "method": "POST",
                "path": "/v1/responses",
                "headers": {"Content-Type": "application/json"},
                "body": base64.b64encode(b'{"input":"hi","stream":true}').decode(),
                "stream": True,
            })
            await asyncio.wait_for(original_first_chunk.wait(), timeout=3.0)
            # Wait a beat to let the buffer have the response.created event.
            for _ in range(50):
                if buffers.get("resp_live") is not None:
                    break
                await asyncio.sleep(0.02)
            assert buffers.get("resp_live") is not None

            # Start a resume tail while the original is still mid-stream.
            await relay.handle_resume({
                "type": FRAME_RESPONSE_RESUME,
                "request_id": "req_resume",
                "response_id": "resp_live",
                "after_cursor": 0,
            })
            # Now release the mock Hermes so the second delta flows.
            gate.set()
            await asyncio.wait_for(original_done.wait(), timeout=3.0)
            await asyncio.wait_for(resume_done.wait(), timeout=3.0)
        finally:
            await relay.close()
    finally:
        await runner.cleanup()

    resume_body = _collect_for("req_resume", frames)
    assert b"response.created" in resume_body
    assert b'"first"' in resume_body
    assert b'"second"' in resume_body
    assert b"response.completed" in resume_body


@pytest.mark.asyncio
async def test_resume_does_not_skip_chunks_appended_during_send():
    """Regression: when the writer appends new chunks while _run_resume's
    inner send loop is awaiting self._send(...), the old `cursor = buf.cursor`
    advance would silently skip those chunks. With cursor += len(pending),
    the next iteration's slice picks them up."""
    from hermes_vylen_gateway.response_buffer import ResponseBufferRegistry

    buffers = ResponseBufferRegistry(grace_seconds=300.0, max_bytes=1 << 20)
    buf = buffers.create("resp_race", 200, {"Content-Type": "text/event-stream"})
    buf.append(b"A")
    buf.append(b"B")
    buf.append(b"C")

    sent: list[dict[str, Any]] = []
    send_started = asyncio.Event()
    end_seen = asyncio.Event()

    async def slow_send(frame):
        # Hold the very first chunk in flight long enough for the writer to
        # append more — that's exactly the race the bug was about.
        if frame["type"] == FRAME_RESPONSE_CHUNK and not send_started.is_set():
            send_started.set()
            await asyncio.sleep(0.05)
        sent.append(frame)
        if frame["type"] == FRAME_RESPONSE_END:
            end_seen.set()

    relay = HermesRelay(slow_send, response_buffers=buffers)
    try:
        await relay.handle_resume({
            "type": FRAME_RESPONSE_RESUME,
            "request_id": "req_race",
            "response_id": "resp_race",
            "after_cursor": 0,
        })
        # Wait for the resume task to begin sending, then append more.
        await asyncio.wait_for(send_started.wait(), timeout=2.0)
        buf.append(b"D")
        buf.append(b"E")
        await asyncio.sleep(0.05)
        buf.finalize()
        await asyncio.wait_for(end_seen.wait(), timeout=2.0)
    finally:
        await relay.close()

    chunks = [
        base64.b64decode(f["data"])
        for f in sent
        if f.get("type") == FRAME_RESPONSE_CHUNK
    ]
    assert chunks == [b"A", b"B", b"C", b"D", b"E"], (
        f"resume dropped chunks under race: got {chunks!r}"
    )
