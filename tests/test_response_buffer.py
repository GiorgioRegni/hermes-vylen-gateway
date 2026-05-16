"""Unit tests for the response buffer registry and SSE id extractor."""

from __future__ import annotations

import asyncio

import pytest

from hermes_vylen_gateway.response_buffer import (
    ResponseBuffer,
    ResponseBufferRegistry,
    ResponseIdExtractor,
)


def test_buffer_appends_and_finalizes():
    buf = ResponseBuffer(response_id="resp_1", status=200, headers={})
    assert buf.cursor == 0
    buf.append(b"hello")
    buf.append(b" world")
    assert buf.cursor == 2
    assert buf.total_bytes == len(b"hello world")
    assert not buf.complete
    buf.finalize()
    assert buf.complete
    assert buf.ended_at is not None


def test_buffer_slice_from_cursor():
    buf = ResponseBuffer(response_id="r", status=200, headers={})
    buf.append(b"a")
    buf.append(b"b")
    buf.append(b"c")
    assert buf.slice_from(0) == [b"a", b"b", b"c"]
    assert buf.slice_from(2) == [b"c"]
    assert buf.slice_from(3) == []
    assert buf.slice_from(99) == []
    assert buf.slice_from(-5) == [b"a", b"b", b"c"]


def test_registry_create_get_drop():
    reg = ResponseBufferRegistry(grace_seconds=10.0, max_bytes=1024)
    buf = reg.create("resp_x", 200, {"content-type": "text/event-stream"})
    assert reg.get("resp_x") is buf
    assert len(reg) == 1
    reg.drop("resp_x")
    assert reg.get("resp_x") is None
    assert len(reg) == 0


def test_registry_sweep_evicts_completed_past_grace():
    reg = ResponseBufferRegistry(grace_seconds=10.0, max_bytes=1024)
    buf = reg.create("r1", 200, {})
    buf.append(b"x")
    # Not complete: shouldn't be evicted regardless of age.
    assert reg.sweep(now=10_000.0) == 0
    buf.finalize()
    fin = buf.ended_at
    assert fin is not None
    # Within grace: kept.
    assert reg.sweep(now=fin + 5.0) == 0
    assert reg.get("r1") is buf
    # Past grace: evicted.
    assert reg.sweep(now=fin + 11.0) == 1
    assert reg.get("r1") is None


def test_registry_sweep_evicts_over_byte_cap():
    reg = ResponseBufferRegistry(grace_seconds=300.0, max_bytes=10)
    buf = reg.create("big", 200, {})
    buf.append(b"x" * 11)
    assert reg.sweep() == 1
    assert reg.get("big") is None


def test_id_extractor_finds_response_created():
    ext = ResponseIdExtractor()
    sse = (
        b'event: response.created\n'
        b'data: {"response":{"id":"resp_abc"}}\n\n'
    )
    rid = ext.feed(sse)
    assert rid == "resp_abc"
    assert ext.response_id == "resp_abc"
    # Subsequent feeds are no-ops.
    assert ext.feed(b"more bytes") is None


def test_id_extractor_handles_split_chunks():
    ext = ResponseIdExtractor()
    sse = b'event: response.created\ndata: {"response":{"id":"resp_split"}}\n\n'
    # Feed it byte-by-byte; should still find the id.
    found = None
    for i in range(len(sse)):
        found = ext.feed(sse[i : i + 1]) or found
    assert found == "resp_split"


def test_id_extractor_ignores_other_events():
    ext = ResponseIdExtractor()
    sse = (
        b'event: response.output_text.delta\n'
        b'data: {"delta":"hello"}\n\n'
        b'event: response.created\n'
        b'data: {"response":{"id":"resp_late"}}\n\n'
    )
    rid = ext.feed(sse)
    assert rid == "resp_late"


def test_id_extractor_silent_when_event_shape_unknown():
    ext = ResponseIdExtractor()
    sse = b'event: something_else\ndata: {"foo":"bar"}\n\n'
    assert ext.feed(sse) is None
    assert ext.response_id is None


def test_id_extractor_accepts_flat_id():
    ext = ResponseIdExtractor()
    sse = b'event: response.created\ndata: {"id":"resp_flat"}\n\n'
    assert ext.feed(sse) == "resp_flat"


def test_id_extractor_bounded_when_id_never_arrives():
    ext = ResponseIdExtractor()
    # Pump 1 MiB of non-matching data; internal buffer must not retain it all.
    junk = b'event: noise\ndata: {"x":1}\n\n' * 4096
    for _ in range(8):
        ext.feed(junk)
    # The internal soft cap is 64 KiB; should be well under that after sweeping.
    assert len(ext._buffer) < 65 * 1024  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_buffer_progressed_event_wakes_waiters():
    buf = ResponseBuffer(response_id="r", status=200, headers={})

    async def waiter() -> bytes:
        await buf.progressed.wait()
        return b"".join(buf.chunks)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    buf.append(b"payload")
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result == b"payload"
