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
