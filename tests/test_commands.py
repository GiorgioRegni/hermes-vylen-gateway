"""Tests for the slash-command enumerator that backs the composer autocomplete.

The enumerator merges three sources (Hermes built-ins, native plugin commands,
installed skills) and the merge order matters: a user-installed skill must be
able to shadow a built-in, since that's what Hermes itself does. These tests
poke each source independently with monkeypatches so we don't need a real
Hermes install to assert behavior.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from hermes_vylen_gateway import commands as commands_module
from hermes_vylen_gateway.commands import (
    CommandSpec,
    CommandsRPC,
    FRAME_COMMANDS_ERROR,
    FRAME_COMMANDS_RESPONSE,
    clear_cache,
    list_available_commands,
)


@dataclasses.dataclass(frozen=True)
class _FakeHermesCmd:
    name: str
    description: str
    aliases: tuple[str, ...] = ()
    args_hint: str = ""


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


def _stub_hermes(monkeypatch, registry, plugin_entries=()):
    """Inject a fake hermes_cli.commands module so the enumerator can run
    without a real Hermes install."""

    monkeypatch.setattr(
        commands_module,
        "_iter_builtin_commands",
        lambda: _emit_builtins(registry, plugin_entries),
    )


def _emit_builtins(registry, plugin_entries):
    for cmd in registry:
        yield CommandSpec(
            command=cmd.name,
            description=cmd.description,
            input_hint=cmd.args_hint,
            aliases=cmd.aliases,
            category="builtin",
        )
    for name, description, args_hint in plugin_entries:
        yield CommandSpec(
            command=name,
            description=description or f"Run /{name}",
            input_hint=args_hint or "",
            category="builtin",
        )


def _stub_skills(monkeypatch, skills):
    monkeypatch.setattr(commands_module, "_iter_skill_commands", lambda: iter(skills))


def test_list_includes_native_builtin_and_skill_commands(monkeypatch):
    _stub_hermes(monkeypatch, [_FakeHermesCmd("new", "Start new", aliases=("reset",))])
    _stub_skills(
        monkeypatch,
        [CommandSpec(command="seo-audit", description="Audit", category="skill")],
    )

    result = {spec.command: spec for spec in list_available_commands(force_refresh=True)}

    # Built-in is present
    assert "new" in result
    assert result["new"].category == "builtin"

    # Skill is present
    assert "seo-audit" in result
    assert result["seo-audit"].category == "skill"

    # Native list contains at least the five we hard-coded
    for cmd in {"status", "reset", "title", "queue", "steer"}:
        assert cmd in result, f"native command {cmd!r} missing"


def test_native_overrides_builtin_on_collision(monkeypatch):
    # Hermes ships `/new` with alias `reset`. Our `_NATIVE_COMMANDS` exports
    # `reset` as a *separate* native command, and the merge order in
    # list_available_commands inserts builtins first then natives — so on the
    # `reset` slug, native wins.
    _stub_hermes(
        monkeypatch,
        [_FakeHermesCmd("reset", "Stub builtin that should be overridden")],
    )
    _stub_skills(monkeypatch, [])

    result = {spec.command: spec for spec in list_available_commands(force_refresh=True)}

    assert result["reset"].category == "native"
    assert "Stub builtin" not in result["reset"].description


def test_skill_overrides_native_on_collision(monkeypatch):
    # Skills win over both builtins and natives — this matches Hermes's own
    # resolution order so a user-installed skill called `/status` can replace
    # the default behavior.
    _stub_hermes(monkeypatch, [])
    _stub_skills(
        monkeypatch,
        [CommandSpec(command="status", description="custom status", category="skill")],
    )

    result = {spec.command: spec for spec in list_available_commands(force_refresh=True)}

    assert result["status"].category == "skill"
    assert result["status"].description == "custom status"


def test_results_are_sorted_alphabetically(monkeypatch):
    _stub_hermes(
        monkeypatch,
        [
            _FakeHermesCmd("zeta", "Last"),
            _FakeHermesCmd("alpha", "First"),
        ],
    )
    _stub_skills(monkeypatch, [])

    names = [spec.command for spec in list_available_commands(force_refresh=True)]

    assert names == sorted(names), "command list should be sorted for deterministic UI"


def test_to_dict_serializes_aliases_and_category():
    spec = CommandSpec(
        command="goal",
        description="Set a goal",
        input_hint="[text]",
        aliases=("g",),
        category="builtin",
    )
    payload = spec.to_dict()
    assert payload == {
        "command": "goal",
        "description": "Set a goal",
        "input_hint": "[text]",
        "aliases": ["g"],
        "category": "builtin",
    }


def test_rpc_replies_with_response_frame():
    """The handler must complete its background task and emit a response
    frame on the supplied send callback, otherwise the cloud HTTP handler
    will time out."""

    sent: list[dict] = []

    async def fake_send(frame):
        sent.append(frame)

    async def scenario():
        rpc = CommandsRPC(fake_send)
        await rpc.handle({"request_id": "req-1"})
        await _drain(rpc)

    asyncio.run(scenario())

    assert len(sent) == 1
    frame = sent[0]
    assert frame["type"] == FRAME_COMMANDS_RESPONSE
    assert frame["request_id"] == "req-1"
    assert "commands" in frame["result"]
    assert isinstance(frame["result"]["commands"], list)


def test_rpc_emits_error_frame_when_enumeration_fails(monkeypatch):
    """If listing throws (e.g. Hermes import explodes), the handler must send
    an error frame so the HTTP client gets a real status instead of hanging
    until the 15s gateway timeout fires."""

    def boom(**_kwargs):
        raise RuntimeError("hermes broke")

    monkeypatch.setattr(commands_module, "list_available_commands", boom)

    sent: list[dict] = []

    async def fake_send(frame):
        sent.append(frame)

    async def scenario():
        rpc = CommandsRPC(fake_send)
        await rpc.handle({"request_id": "req-err"})
        await _drain(rpc)

    asyncio.run(scenario())

    assert len(sent) == 1
    assert sent[0]["type"] == FRAME_COMMANDS_ERROR
    assert sent[0]["request_id"] == "req-err"
    assert "hermes broke" in sent[0]["message"]


def test_rpc_ignores_frames_without_request_id():
    """A missing request_id means the cloud could never route the response;
    silently dropping is safer than sending a reply to nobody."""

    sent: list[dict] = []

    async def fake_send(frame):
        sent.append(frame)

    async def scenario():
        rpc = CommandsRPC(fake_send)
        await rpc.handle({})
        await _drain(rpc)

    asyncio.run(scenario())
    assert sent == []


async def _drain(rpc: CommandsRPC) -> None:
    """Wait for every in-flight handler task to finish naturally. Calling
    rpc.close() instead would *cancel* them — which is the right behavior on
    shutdown but kills the tests before the response frame is sent."""

    tasks = list(rpc._tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
