from __future__ import annotations

import asyncio

import pytest

from hermes_vylen_gateway.adapter import _sweep_loop
from hermes_vylen_gateway.event_log import EventLogRegistry, ResumeExpired, RetainedEventLog


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_replay_after_cursor_and_independent_client_cursors():
    registry = EventLogRegistry()
    log = registry.get_or_create("chat_a")
    first = log.append("message.created", {"text": "one"})
    second = log.append("message.created", {"text": "two"})

    assert [event.seq for event in log.replay_after(0)] == [first.seq, second.seq]
    assert [event.seq for event in log.replay_after(first.seq)] == [second.seq]

    registry.acknowledge("chat_a", "client_phone", first.seq)
    registry.acknowledge("chat_a", "client_tablet", second.seq)

    assert registry.cursor("chat_a", "client_phone") == first.seq
    assert registry.cursor("chat_a", "client_tablet") == second.seq
    assert registry.cursor("chat_b", "client_phone") == 0


def test_count_eviction_expires_old_cursor():
    log = RetainedEventLog("chat_a", max_events=2)
    log.append("event", {"n": 1})
    log.append("event", {"n": 2})
    log.append("event", {"n": 3})

    assert [event.seq for event in log.events] == [2, 3]
    assert log.floor_seq == 1
    with pytest.raises(ResumeExpired) as exc:
        log.replay_after(0)
    assert exc.value.floor_seq == 1
    assert exc.value.latest_seq == 3


def test_ttl_eviction_expires_old_cursor():
    clock = Clock()
    log = RetainedEventLog("chat_a", ttl_seconds=5, now=clock)
    log.append("event", {"n": 1})
    clock.advance(6)
    log.append("event", {"n": 2})

    assert [event.seq for event in log.events] == [2]
    with pytest.raises(ResumeExpired):
        log.replay_after(0)


@pytest.mark.asyncio
async def test_future_cursor_expires_on_fresh_log_when_requested():
    log = RetainedEventLog("chat_a")

    with pytest.raises(ResumeExpired) as exc:
        async for _ in log.tail_after(184, reject_future_cursor=True):
            pass

    assert exc.value.floor_seq == 0
    assert exc.value.latest_seq == 0


@pytest.mark.asyncio
async def test_future_cursor_expires_on_active_log_when_requested():
    log = RetainedEventLog("chat_a")
    log.append("event", {"n": 1})

    with pytest.raises(ResumeExpired) as exc:
        async for _ in log.tail_after(184, reject_future_cursor=True):
            pass

    assert exc.value.floor_seq == 0
    assert exc.value.latest_seq == 1


def test_byte_eviction_bounds_retained_payloads():
    log = RetainedEventLog("chat_a", max_bytes=70)
    log.append("event", {"text": "a" * 30})
    log.append("event", {"text": "b" * 30})

    assert [event.seq for event in log.events] == [2]
    assert log.total_bytes <= 70


def test_registry_sweep_drops_expired_empty_logs_and_cursors():
    clock = Clock()
    registry = EventLogRegistry(ttl_seconds=5, now=clock)
    log = registry.get_or_create("chat_a")
    event = log.append("event", {"n": 1})
    registry.acknowledge("chat_a", "client_phone", event.seq)

    clock.advance(6)
    assert registry.sweep() == 1

    assert registry.get("chat_a") is None
    assert registry.cursor("chat_a", "client_phone") == 0


@pytest.mark.asyncio
async def test_adapter_sweep_loop_uses_event_log_registry_sweep():
    clock = Clock()
    registry = EventLogRegistry(ttl_seconds=1, now=clock)
    registry.get_or_create("chat_a").append("event", {"n": 1})
    clock.advance(2)

    task = asyncio.create_task(_sweep_loop(registry, interval_seconds=0.01, label="chat event log"))
    try:
        for _ in range(50):
            if registry.get("chat_a") is None:
                break
            await asyncio.sleep(0.01)
        assert registry.get("chat_a") is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_live_tailers_receive_same_events_without_consuming_globally():
    log = RetainedEventLog("chat_a")
    phone: list[int] = []
    tablet: list[int] = []

    async def collect(out: list[int]) -> None:
        async for event in log.tail_after(0):
            out.append(event.seq)

    phone_task = asyncio.create_task(collect(phone))
    tablet_task = asyncio.create_task(collect(tablet))
    await asyncio.sleep(0)

    log.append("message.created", {"text": "one"})
    log.append("message.created", {"text": "two"})
    await asyncio.sleep(0)
    log.close()

    await asyncio.wait_for(asyncio.gather(phone_task, tablet_task), timeout=1.0)

    assert phone == [1, 2]
    assert tablet == [1, 2]
