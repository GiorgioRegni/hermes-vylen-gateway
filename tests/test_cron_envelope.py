"""Unit tests for the Vylen gateway adapter's lightweight helpers.

The plugin's `send()` strips
Hermes's `Cronjob Response: …\\n(job_id: …)\\n---\\n…` chrome so the Vylen
UI sees the LLM's actual output and gets job_id/name as structured
metadata. Tracked upstream-ask: get this as a metadata dict instead.
"""

import asyncio

import hermes_vylen_gateway.adapter as adapter_mod
from hermes_vylen_gateway.adapter import (
    VYLEN_INBOX_CHAT_ID,
    _parse_cron_envelope,
    _push_cursor_chat_id,
)


def test_envelope_with_footer():
    raw = (
        "Cronjob Response: vylen-smoke\n"
        "(job_id: 7f9b42f1998a)\n"
        "-------------\n"
        "\n"
        "22:35:01 7842 smoke\n"
        "\n"
        "To stop or manage this job, send me a new message (e.g. \"stop reminder vylen-smoke\")."
    )
    body, job_id, name = _parse_cron_envelope(raw)
    assert body == "22:35:01 7842 smoke"
    assert job_id == "7f9b42f1998a"
    assert name == "vylen-smoke"


def test_envelope_without_footer():
    raw = (
        "Cronjob Response: weather\n"
        "(job_id: abc123)\n"
        "-------------\n"
        "\n"
        "It's sunny."
    )
    body, job_id, name = _parse_cron_envelope(raw)
    assert body == "It's sunny."
    assert job_id == "abc123"
    assert name == "weather"


def test_envelope_multiline_body():
    raw = (
        "Cronjob Response: ledger\n"
        "(job_id: deadbeef)\n"
        "-------------\n"
        "\n"
        "Line one.\n"
        "Line two.\n"
        "\n"
        "Line four.\n"
        "\n"
        "To stop or manage this job, send me a new message."
    )
    body, job_id, name = _parse_cron_envelope(raw)
    assert body == "Line one.\nLine two.\n\nLine four."
    assert job_id == "deadbeef"
    assert name == "ledger"


def test_non_envelope_passes_through():
    """Plain chat sends shouldn't be mangled by the cron parser."""
    raw = "Hello! How can I help?"
    body, job_id, name = _parse_cron_envelope(raw)
    assert body == raw
    assert job_id == ""
    assert name == ""


def test_partial_envelope_passes_through():
    """If only the prefix matches but the dashed separator is missing,
    treat it as ordinary text rather than partially stripping."""
    raw = "Cronjob Response: surprise\nbut no job id line"
    body, job_id, name = _parse_cron_envelope(raw)
    assert body == raw
    assert job_id == ""
    assert name == ""


def test_push_cursor_chat_id_uses_vylen_inbox_bucket():
    assert _push_cursor_chat_id("custom_home") == VYLEN_INBOX_CHAT_ID
    assert _push_cursor_chat_id("") == VYLEN_INBOX_CHAT_ID
    assert _push_cursor_chat_id(None) == VYLEN_INBOX_CHAT_ID


def test_adapter_keeps_blob_registry_across_socket_teardown(monkeypatch):
    class FakeBasePlatformAdapter:
        def __init__(self, config, platform):
            self.config = config
            self.platform = platform

    class FakePlatform:
        def __init__(self, name):
            self.name = name

    class FakeClient:
        async def close(self):
            return None

    monkeypatch.setattr(
        adapter_mod,
        "_import_hermes",
        lambda: (FakeBasePlatformAdapter, FakePlatform),
    )
    adapter_cls = adapter_mod.make_adapter_class()
    adapter = adapter_cls(config={})
    blobs = adapter._blobs
    adapter._client = FakeClient()

    asyncio.run(adapter._teardown_session())

    assert adapter._blobs is blobs
