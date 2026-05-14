"""Unit tests for the cron-envelope parser. The plugin's `send()` strips
Hermes's `Cronjob Response: …\\n(job_id: …)\\n---\\n…` chrome so the Vylen
UI sees the LLM's actual output and gets job_id/name as structured
metadata. Tracked upstream-ask: get this as a metadata dict instead.
"""

from hermes_vylen_gateway.adapter import _parse_cron_envelope


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
