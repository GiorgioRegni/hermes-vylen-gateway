from __future__ import annotations

import types

import pytest

from hermes_vylen_gateway import health


@pytest.mark.asyncio
async def test_health_redacts_chat_state_paths_and_raw_messages(monkeypatch):
    frames: list[dict] = []

    async def send(frame: dict) -> None:
        frames.append(dict(frame))

    monkeypatch.setattr(health, "check_available", lambda: (True, None))
    reporter = health.HealthReporter(
        send,
        chat_state_status=lambda: types.SimpleNamespace(
            status="open_failed",
            message="chat state path is not writable: /home/user/.hermes/vylen/chat-state.sqlite3",
            quarantined_path="/home/user/.hermes/vylen/chat-state.sqlite3.corrupt-20260522T120000Z",
        ),
    )

    await reporter._probe_and_send()

    frame = frames[-1]
    assert frame["chat_state_status"] == "open_failed"
    assert frame["chat_state_message"] == "Local Vylen chat state is unavailable."
    assert frame["last_error"] == "chat_state: open_failed"
    assert "chat_state_quarantined_path" not in frame
    assert ".hermes" not in str(frame)


@pytest.mark.asyncio
async def test_health_frame_includes_resource_pressure_buckets(monkeypatch):
    frames: list[dict] = []

    async def send(frame: dict) -> None:
        frames.append(dict(frame))

    monkeypatch.setattr(health, "check_available", lambda: (True, None))
    reporter = health.HealthReporter(
        send,
        resource_sampler=types.SimpleNamespace(sample=lambda: {"cpu": "warning", "memory": "ok"}),
    )

    await reporter._probe_and_send()

    assert frames[-1]["resource_pressure"] == {"cpu": "warning", "memory": "ok"}
    assert "80" not in str(frames[-1])


@pytest.mark.asyncio
async def test_resource_sampler_failure_does_not_block_health(monkeypatch):
    frames: list[dict] = []

    async def send(frame: dict) -> None:
        frames.append(dict(frame))

    def fail():
        raise RuntimeError("metrics unavailable")

    monkeypatch.setattr(health, "check_available", lambda: (True, None))
    reporter = health.HealthReporter(
        send,
        resource_sampler=types.SimpleNamespace(sample=fail),
    )

    await reporter._probe_and_send()

    assert frames[-1]["hermes_status"] == "ok"
    assert frames[-1]["resource_pressure"] == {"cpu": "unknown", "memory": "unknown"}
    assert "metrics unavailable" not in str(frames[-1])
