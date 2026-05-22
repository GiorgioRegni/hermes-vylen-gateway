"""Slash-command enumeration for the Vylen UI autocomplete.

Combines three sources into a single ordered list:

* **native** — commands handled directly by this gateway plugin
  (`_native_chat_command` in :mod:`adapter`).
* **builtin** — Hermes's structured ``COMMAND_REGISTRY`` plus plugin-registered
  slash commands, surfaced through ``hermes_cli.commands``.
* **skill** — installed Hermes skills under ``~/.hermes/skills/`` and external
  skill dirs, surfaced through ``agent.skill_commands.get_skill_commands``.

Later sources take precedence on key collision (so a user-installed skill can
shadow a built-in command, matching Hermes's own resolution order).

Results are cached for ~30 seconds so a fresh skill install shows up on the
next composer focus without restart, while back-to-back fetches don't rescan
the filesystem.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)


FRAME_COMMANDS_REQUEST = "commands_request"
FRAME_COMMANDS_RESPONSE = "commands_response"
FRAME_COMMANDS_ERROR = "commands_error"

CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class CommandSpec:
    """A single slash command exposed to the UI.

    Mirrors the ACP ``AvailableCommand`` shape (see hermes-agent
    ``acp_adapter/server.py``) so a future move to ACP-sourced commands is
    a serialization swap, not a UI rewrite.
    """

    command: str               # canonical name without leading slash, e.g. "goal"
    description: str
    input_hint: str = ""       # e.g. "[text | pause | resume | clear | status]"
    aliases: tuple[str, ...] = ()
    category: str = "builtin"  # "native" | "builtin" | "skill"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d


_NATIVE_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        command="status",
        description="Show session info",
        category="native",
    ),
    CommandSpec(
        command="reset",
        description="Start a new chat session",
        category="native",
    ),
    CommandSpec(
        command="title",
        description="Set a title for the current chat",
        input_hint="<name>",
        category="native",
    ),
    CommandSpec(
        command="queue",
        description="Queue a prompt for the next turn (doesn't interrupt)",
        input_hint="<prompt>",
        aliases=("q",),
        category="native",
    ),
    CommandSpec(
        command="steer",
        description="Inject a message after the next tool call without interrupting",
        input_hint="<prompt>",
        category="native",
    ),
)


_cache: tuple[float, list[CommandSpec]] | None = None


def list_available_commands(*, force_refresh: bool = False) -> list[CommandSpec]:
    """Return the merged list of commands. Cached for :data:`CACHE_TTL_SECONDS`.

    `force_refresh=True` bypasses the cache (used by tests; production callers
    should rely on the TTL).
    """

    global _cache
    now = time.monotonic()
    if not force_refresh and _cache is not None:
        cached_at, cached = _cache
        if now - cached_at < CACHE_TTL_SECONDS:
            return list(cached)

    by_command: dict[str, CommandSpec] = {}

    for spec in _iter_builtin_commands():
        by_command[spec.command] = spec
    for spec in _NATIVE_COMMANDS:
        by_command[spec.command] = spec
    for spec in _iter_skill_commands():
        by_command[spec.command] = spec

    result = sorted(by_command.values(), key=lambda c: c.command)
    _cache = (now, result)
    return list(result)


def clear_cache() -> None:
    """Reset the in-process cache. Tests call this between cases."""

    global _cache
    _cache = None


def _iter_builtin_commands() -> Iterable[CommandSpec]:
    """Yield CommandSpec from Hermes's structured COMMAND_REGISTRY.

    Filters out cli_only entries (they don't make sense over the gateway)
    using Hermes's own ``_is_gateway_available`` helper. Plugin-registered
    commands from ``hermes_cli.plugins`` are included via
    ``_iter_plugin_command_entries`` for parity with what users see in
    Hermes's own ``/commands`` output.
    """

    try:
        from hermes_cli.commands import (  # type: ignore[import-not-found]
            COMMAND_REGISTRY,
            _is_gateway_available,
            _iter_plugin_command_entries,
            _resolve_config_gates,
        )
    except Exception:
        logger.warning("vylen commands: hermes_cli.commands unavailable", exc_info=True)
        return

    try:
        overrides = _resolve_config_gates()
    except Exception:
        overrides = set()

    for cmd in COMMAND_REGISTRY:
        try:
            if not _is_gateway_available(cmd, overrides):
                continue
            aliases = tuple(
                a for a in getattr(cmd, "aliases", ())
                if a.replace("-", "_") != cmd.name.replace("-", "_")
            )
            yield CommandSpec(
                command=cmd.name,
                description=cmd.description,
                input_hint=cmd.args_hint or "",
                aliases=aliases,
                category="builtin",
            )
        except Exception:
            logger.debug("vylen commands: skipping registry entry %r", cmd, exc_info=True)

    try:
        plugin_entries = _iter_plugin_command_entries()
    except Exception:
        plugin_entries = []
    for name, description, args_hint in plugin_entries:
        yield CommandSpec(
            command=name,
            description=description or f"Run /{name}",
            input_hint=args_hint or "",
            category="builtin",
        )


def _iter_skill_commands() -> Iterable[CommandSpec]:
    """Yield CommandSpec for each installed Hermes skill."""

    try:
        from agent.skill_commands import get_skill_commands  # type: ignore[import-not-found]
    except Exception:
        logger.debug("vylen commands: agent.skill_commands unavailable", exc_info=True)
        return

    try:
        skills = get_skill_commands() or {}
    except Exception:
        logger.warning("vylen commands: scanning skills failed", exc_info=True)
        return

    for key, info in skills.items():
        if not isinstance(key, str) or not key.startswith("/"):
            continue
        command = key.lstrip("/").strip()
        if not command:
            continue
        description = ""
        if isinstance(info, dict):
            description = str(info.get("description") or "").strip()
        yield CommandSpec(
            command=command,
            description=description or f"Invoke the {command} skill",
            input_hint="",
            category="skill",
        )


class CommandsRPC:
    """Handles ``FRAME_COMMANDS_REQUEST`` frames from Vylen Cloud.

    The cloud answers ``GET /v1/instances/{id}/commands`` by forwarding the
    request to this handler. Reply latency is bounded by the cached enumerator
    so back-to-back requests don't rescan the filesystem.
    """

    def __init__(self, send_frame: Callable[[dict[str, Any]], Awaitable[None]]):
        self._send = send_frame
        self._tasks: set[asyncio.Task] = set()

    async def handle(self, frame: dict[str, Any]) -> None:
        request_id = str(frame.get("request_id") or "")
        if not request_id:
            logger.warning("commands rpc: request frame missing request_id")
            return
        task = asyncio.create_task(self._run(request_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _run(self, request_id: str) -> None:
        try:
            specs = list_available_commands()
        except Exception as exc:  # noqa: BLE001
            logger.exception("commands rpc: enumeration failed")
            await self._send({
                "type": FRAME_COMMANDS_ERROR,
                "request_id": request_id,
                "code": "COMMANDS_UNAVAILABLE",
                "message": str(exc),
            })
            return
        await self._send({
            "type": FRAME_COMMANDS_RESPONSE,
            "request_id": request_id,
            "result": {
                "commands": [spec.to_dict() for spec in specs],
            },
        })
