"""Read-only Hermes memory control-plane RPCs.

The Cloud side exposes typed memory routes; this module answers those calls
inside the Hermes process without turning the normal HTTP relay into a local
file browser.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

FRAME_MEMORY_REQUEST = "memory_request"
FRAME_MEMORY_RESPONSE = "memory_response"
FRAME_MEMORY_ERROR = "memory_error"

ENTRY_DELIMITER = "\n§\n"
SESSION_WARNING = (
    "Memory edits are saved now, but Hermes applies them to system prompt "
    "context on the next session start."
)

TARGETS = {
    "memory": {
        "filename": "MEMORY.md",
        "label": "Core memory",
        "enabled_key": "memory_enabled",
        "limit_key": "memory_char_limit",
        "default_limit": 2200,
    },
    "user": {
        "filename": "USER.md",
        "label": "User profile",
        "enabled_key": "user_profile_enabled",
        "limit_key": "user_char_limit",
        "default_limit": 1375,
    },
}


class MemoryRPC:
    """Handles read-only memory RPC frames from Vylen Cloud."""

    def __init__(self, send_frame: Callable[[dict[str, Any]], Awaitable[None]]):
        self._send = send_frame
        self._tasks: set[asyncio.Task] = set()

    async def handle(self, frame: dict[str, Any]) -> None:
        request_id = frame.get("request_id") or ""
        if not request_id:
            logger.warning("memory rpc: request frame missing request_id")
            return
        task = asyncio.create_task(self._run(request_id, frame))
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

    async def _run(self, request_id: str, frame: dict[str, Any]) -> None:
        rpc = frame.get("rpc") or ""
        try:
            if rpc == "memory.status":
                result = build_memory_status(include_entries=False)
            elif rpc == "memory.core.read":
                result = build_memory_status(include_entries=True)
            elif rpc == "memory.providers.status":
                result = build_provider_status()
            else:
                await self._send_error(request_id, "BAD_MEMORY_RPC", f"Unknown memory RPC '{rpc}'")
                return
            await self._send({
                "type": FRAME_MEMORY_RESPONSE,
                "request_id": request_id,
                "rpc": rpc,
                "result": result,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("memory rpc failed: %s", rpc)
            await self._send_error(request_id, "MEMORY_UNAVAILABLE", str(exc))

    async def _send_error(self, request_id: str, code: str, message: str) -> None:
        await self._send({
            "type": FRAME_MEMORY_ERROR,
            "request_id": request_id,
            "code": code,
            "message": message,
        })


def build_memory_status(*, include_entries: bool) -> dict[str, Any]:
    cfg, config_available = _load_memory_config()
    memory_dir = _memory_dir()
    targets = {
        target: _target_status(
            target,
            memory_dir / meta["filename"],
            cfg,
            config_available=config_available,
            include_entries=include_entries,
        )
        for target, meta in TARGETS.items()
    }
    return {
        "mode": "file",
        "memory_dir": str(memory_dir),
        "targets": targets,
        "session_warning": SESSION_WARNING,
    }


def build_provider_status() -> dict[str, Any]:
    cfg, config_available = _load_memory_config()
    provider = str(cfg.get("provider") or "").strip()
    active = bool(provider)
    health = "ok" if active else "unconfigured"
    return {
        "provider": provider or "builtin",
        "active": active,
        "available": active,
        "storage_class": "external" if active else "builtin",
        "capabilities": {
            "semanticSearch": False,
            "sessionSync": False,
            "memoryWriteMirror": False,
        },
        "health": {
            "status": health,
            "checked_at": _now_iso(),
            "error": None if config_available else "Hermes config unavailable; provider status is best-effort.",
        },
        "required_env": [],
        "session_warning": SESSION_WARNING,
    }


def _target_status(
    target: str,
    path: Path,
    cfg: dict[str, Any],
    *,
    config_available: bool,
    include_entries: bool,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"invalid memory target: {target}")
    meta = TARGETS[target]
    limit = _int_config(cfg.get(meta["limit_key"]), meta["default_limit"])
    enabled = bool(cfg.get(meta["enabled_key"], False))
    base: dict[str, Any] = {
        "target": target,
        "label": meta["label"],
        "filename": meta["filename"],
        "enabled": enabled,
        "enabled_source": "config" if config_available else "unknown",
        "char_limit": limit,
        "char_count": 0,
        "entry_count": 0,
        "capacity_state": "ok",
        "status": "missing",
        "revision_hash": "",
        "mtime": None,
        "error": None,
    }
    if include_entries:
        base["entries"] = []

    if not path.exists():
        return base

    try:
        raw_bytes = path.read_bytes()
        raw = raw_bytes.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        base["status"] = "unreadable"
        base["capacity_state"] = "invalid"
        base["error"] = str(exc)
        return base

    entries = _parse_entries(raw)
    char_count = len(ENTRY_DELIMITER.join(entries)) if entries else 0
    base.update({
        "char_count": char_count,
        "entry_count": len(entries),
        "capacity_state": _capacity_state(char_count, limit),
        "status": _target_health(raw, char_count, limit),
        "revision_hash": hashlib.sha256(raw_bytes).hexdigest(),
        "mtime": _mtime_iso(path),
    })
    if include_entries:
        base["entries"] = [
            {"index": i, "content": entry, "char_count": len(entry)}
            for i, entry in enumerate(entries)
        ]
    return base


def _target_health(raw: str, char_count: int, limit: int) -> str:
    if raw.strip() == "":
        return "empty"
    if limit > 0 and char_count > limit:
        return "over_capacity"
    return "readable"


def _parse_entries(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]


def _capacity_state(char_count: int, limit: int) -> str:
    if limit <= 0:
        return "invalid"
    pct = char_count / limit
    if pct >= 0.95:
        return "full"
    if pct >= 0.80:
        return "near_capacity"
    if pct >= 0.70:
        return "watch"
    return "ok"


def _memory_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "memories"
    except Exception:  # noqa: BLE001
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "memories"


def _load_memory_config() -> tuple[dict[str, Any], bool]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        mem = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        return (mem if isinstance(mem, dict) else {}), True
    except Exception:  # noqa: BLE001
        logger.debug("memory rpc: Hermes config unavailable", exc_info=True)
        return {}, False


def _int_config(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mtime_iso(path: Path) -> str | None:
    try:
        return _iso_from_timestamp(path.stat().st_mtime)
    except OSError:
        return None


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_from_timestamp(value: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(value, _dt.timezone.utc).isoformat().replace("+00:00", "Z")
