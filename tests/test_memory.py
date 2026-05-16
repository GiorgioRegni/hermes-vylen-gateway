from __future__ import annotations

import pytest

from hermes_vylen_gateway.memory import (
    ENTRY_DELIMITER,
    _capacity_state,
    _target_status,
    build_memory_status,
)


def test_memory_status_handles_missing_and_empty_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "USER.md").write_text("", encoding="utf-8")

    result = build_memory_status(include_entries=True)

    memory = result["targets"]["memory"]
    user = result["targets"]["user"]
    assert memory["status"] == "missing"
    assert memory["entries"] == []
    assert user["status"] == "empty"
    assert user["entries"] == []


def test_memory_status_parses_multiline_entries_without_splitting_inline_section_sign(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text(
        "First line\nkeeps going § inline" + ENTRY_DELIMITER + "Second entry",
        encoding="utf-8",
    )

    result = build_memory_status(include_entries=True)
    entries = result["targets"]["memory"]["entries"]

    assert [e["content"] for e in entries] == [
        "First line\nkeeps going § inline",
        "Second entry",
    ]
    assert result["targets"]["memory"]["entry_count"] == 2
    assert result["targets"]["memory"]["status"] == "readable"


@pytest.mark.parametrize(
    ("chars", "limit", "want"),
    [
        (69, 100, "ok"),
        (70, 100, "watch"),
        (80, 100, "near_capacity"),
        (95, 100, "full"),
        (1, 0, "invalid"),
    ],
)
def test_capacity_thresholds(chars, limit, want):
    assert _capacity_state(chars, limit) == want


def test_over_capacity_status(monkeypatch, tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text("abcdef", encoding="utf-8")

    status = _target_status(
        "memory",
        path,
        {"memory_enabled": True, "memory_char_limit": 3},
        config_available=True,
        include_entries=False,
    )

    assert status["status"] == "over_capacity"
    assert status["capacity_state"] == "full"


def test_target_status_rejects_unknown_target(tmp_path):
    with pytest.raises(ValueError):
        _target_status(
            "soul",
            tmp_path / "SOUL.md",
            {},
            config_available=False,
            include_entries=False,
        )
