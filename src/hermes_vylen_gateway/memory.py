"""Hermes memory control-plane RPCs.

The Cloud side exposes typed memory routes; this module answers those calls
inside the Hermes process without turning the normal HTTP relay into a local
file browser.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback, Hermes hosts are usually Unix
    fcntl = None

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

_THREAT_PATTERNS = [
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private_key"),
    (r"\b(Bearer|api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_\-\.]{16,}", "credential_like"),
]

_INVISIBLE_CHARS = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
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
            elif rpc == "memory.core.preview":
                result = preview_memory_write(_parse_params(frame))
            elif rpc == "memory.core.write":
                result = write_memory(_parse_params(frame))
            elif rpc == "memory.snapshots.list":
                result = list_memory_snapshots(_parse_params(frame))
            elif rpc == "memory.snapshots.create":
                result = create_memory_snapshot(_parse_params(frame))
            elif rpc == "memory.snapshots.restore":
                result = restore_memory_snapshot(_parse_params(frame))
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
        except MemoryRPCError as exc:
            await self._send_error(request_id, exc.code, exc.message)
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


def preview_memory_write(params: dict[str, Any]) -> dict[str, Any]:
    target, expected_hash, ops, _reason = _parse_write_request(params)
    cfg, config_available = _load_memory_config()
    path = _path_for_target(target)
    before_raw = _read_target_text(path)
    before_entries = _parse_entries(before_raw)
    previous_hash = _hash_text(before_raw)
    if expected_hash != previous_hash:
        raise MemoryRPCError("MEMORY_REVISION_CONFLICT", "Memory changed since it was loaded.")

    after_entries = _apply_ops(before_entries, ops)
    _validate_entries(target, after_entries, cfg)
    after_raw = _render_entries(after_entries)
    return _preview_response(
        target,
        before_entries,
        after_entries,
        previous_hash,
        _hash_text(after_raw),
        cfg,
        config_available=config_available,
    )


def write_memory(params: dict[str, Any]) -> dict[str, Any]:
    target, expected_hash, ops, reason = _parse_write_request(params)
    cfg, config_available = _load_memory_config()
    path = _path_for_target(target)

    with _file_lock(path):
        before_raw = _read_target_text(path)
        before_entries = _parse_entries(before_raw)
        previous_hash = _hash_text(before_raw)
        if expected_hash != previous_hash:
            raise MemoryRPCError("MEMORY_REVISION_CONFLICT", "Memory changed since it was loaded.")

        after_entries = _apply_ops(before_entries, ops)
        _validate_entries(target, after_entries, cfg)
        after_raw = _render_entries(after_entries)
        snapshot_id = _create_snapshot(target, path, before_raw, previous_hash, reason)
        _atomic_write_text(path, after_raw)
        verified = _read_target_text(path)
        new_hash = _hash_text(verified)
        if new_hash != _hash_text(after_raw):
            raise MemoryRPCError("MEMORY_WRITE_VERIFY_FAILED", "Memory write verification failed.")

    response = _preview_response(
        target,
        before_entries,
        after_entries,
        previous_hash,
        new_hash,
        cfg,
        config_available=config_available,
    )
    response["snapshot_id"] = snapshot_id
    response["session_warning"] = SESSION_WARNING
    return response


def list_memory_snapshots(params: dict[str, Any] | None = None) -> dict[str, Any]:
    target_filter = ""
    if params:
        raw_target = str(params.get("target") or "")
        if raw_target:
            if raw_target not in TARGETS:
                raise MemoryRPCError("MEMORY_BAD_TARGET", "target must be 'memory' or 'user'.")
            target_filter = raw_target

    snapshots = []
    targets = [target_filter] if target_filter else list(TARGETS)
    for target in targets:
        snapshot_dir = _snapshot_dir(target)
        if not snapshot_dir.exists():
            continue
        for meta_path in snapshot_dir.glob("*.json"):
            metadata = _read_snapshot_metadata(meta_path)
            if metadata is None:
                continue
            body_path = snapshot_dir / f"{metadata['id']}.md"
            metadata["available"] = body_path.exists()
            snapshots.append(metadata)

    snapshots.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {
        "snapshots": snapshots,
        "session_warning": SESSION_WARNING,
    }


def create_memory_snapshot(params: dict[str, Any]) -> dict[str, Any]:
    target = _parse_target(params)
    expected_hash = str(params.get("expected_revision_hash") or "")
    reason = str(params.get("reason") or "manual snapshot").strip()
    path = _path_for_target(target)

    with _file_lock(path):
        text = _read_target_text(path)
        revision_hash = _hash_text(text)
        if expected_hash and expected_hash != revision_hash:
            raise MemoryRPCError("MEMORY_REVISION_CONFLICT", "Memory changed since it was loaded.")
        snapshot_id = _create_snapshot(target, path, text, revision_hash, reason)

    return {
        "snapshot": _snapshot_metadata_by_id(target, snapshot_id),
        "session_warning": SESSION_WARNING,
    }


def restore_memory_snapshot(params: dict[str, Any]) -> dict[str, Any]:
    target = _parse_target(params)
    snapshot_id = str(params.get("snapshot_id") or "").strip()
    if not _valid_snapshot_id(snapshot_id):
        raise MemoryRPCError("MEMORY_BAD_SNAPSHOT", "snapshot_id is invalid.")
    expected_hash = str(params.get("expected_revision_hash") or "")
    reason = str(params.get("reason") or f"restore {snapshot_id}").strip()
    cfg, config_available = _load_memory_config()
    path = _path_for_target(target)
    snapshot = _snapshot_metadata_by_id(target, snapshot_id)
    if snapshot.get("target") != target:
        raise MemoryRPCError("MEMORY_BAD_SNAPSHOT", "snapshot target does not match request target.")
    snapshot_body = _read_snapshot_body(target, snapshot_id)
    snapshot_entries = _parse_entries(snapshot_body)
    _validate_entries(target, snapshot_entries, cfg)

    with _file_lock(path):
        before_raw = _read_target_text(path)
        before_entries = _parse_entries(before_raw)
        previous_hash = _hash_text(before_raw)
        if expected_hash and expected_hash != previous_hash:
            raise MemoryRPCError("MEMORY_REVISION_CONFLICT", "Memory changed since it was loaded.")
        rollback_snapshot_id = _create_snapshot(
            target,
            path,
            before_raw,
            previous_hash,
            f"rollback before restoring {snapshot_id}: {reason}",
        )
        _atomic_write_text(path, snapshot_body)
        verified = _read_target_text(path)
        new_hash = _hash_text(verified)
        if new_hash != _hash_text(snapshot_body):
            raise MemoryRPCError("MEMORY_WRITE_VERIFY_FAILED", "Memory restore verification failed.")

    response = _preview_response(
        target,
        before_entries,
        snapshot_entries,
        previous_hash,
        new_hash,
        cfg,
        config_available=config_available,
    )
    response["snapshot_id"] = snapshot_id
    response["restored_snapshot"] = snapshot
    response["rollback_snapshot_id"] = rollback_snapshot_id
    response["session_warning"] = SESSION_WARNING
    return response


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


class MemoryRPCError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _parse_params(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("params") or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw) if raw else {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _parse_target(params: dict[str, Any]) -> str:
    target = str(params.get("target") or "")
    if target not in TARGETS:
        raise MemoryRPCError("MEMORY_BAD_TARGET", "target must be 'memory' or 'user'.")
    return target


def _parse_write_request(params: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]], str]:
    target = _parse_target(params)
    expected_hash = str(params.get("expected_revision_hash") or "")
    ops = params.get("ops")
    if not isinstance(ops, list) or not ops:
        raise MemoryRPCError("MEMORY_BAD_REQUEST", "ops must be a non-empty array.")
    typed_ops = [op for op in ops if isinstance(op, dict)]
    if len(typed_ops) != len(ops):
        raise MemoryRPCError("MEMORY_BAD_REQUEST", "Every op must be an object.")
    reason = str(params.get("reason") or "").strip()
    return target, expected_hash, typed_ops, reason


def _apply_ops(entries: list[str], ops: list[dict[str, Any]]) -> list[str]:
    next_entries = list(entries)
    for op in ops:
        typ = str(op.get("type") or "")
        if typ == "add":
            content = str(op.get("content") or "").strip()
            if not content:
                raise MemoryRPCError("MEMORY_BAD_REQUEST", "add.content cannot be empty.")
            next_entries.append(content)
        elif typ == "replace":
            idx = _op_index(op, "index", len(next_entries))
            content = str(op.get("content") or "").strip()
            if not content:
                raise MemoryRPCError("MEMORY_BAD_REQUEST", "replace.content cannot be empty.")
            next_entries[idx] = content
        elif typ == "remove":
            idx = _op_index(op, "index", len(next_entries))
            next_entries.pop(idx)
        elif typ == "reorder":
            from_idx = _op_index(op, "from_index", len(next_entries))
            to_idx = _op_index(op, "to_index", len(next_entries), allow_end=True)
            item = next_entries.pop(from_idx)
            if to_idx > from_idx:
                to_idx -= 1
            next_entries.insert(to_idx, item)
        else:
            raise MemoryRPCError("MEMORY_BAD_OPERATION", f"Unsupported memory op '{typ}'.")
    return next_entries


def _op_index(op: dict[str, Any], key: str, length: int, *, allow_end: bool = False) -> int:
    try:
        idx = int(op.get(key))
    except (TypeError, ValueError):
        raise MemoryRPCError("MEMORY_BAD_REQUEST", f"{key} must be an integer.") from None
    upper = length if allow_end else length - 1
    if idx < 0 or idx > upper:
        raise MemoryRPCError("MEMORY_BAD_REQUEST", f"{key} is out of range.")
    return idx


def _validate_entries(target: str, entries: list[str], cfg: dict[str, Any]) -> None:
    seen = set()
    for entry in entries:
        if entry in seen:
            raise MemoryRPCError("MEMORY_DUPLICATE_ENTRY", "Exact duplicate memory entries are not allowed.")
        seen.add(entry)
        risk = _scan_entry(entry)
        if risk:
            raise MemoryRPCError("MEMORY_RISK_BLOCKED", risk)
    limit = _char_limit(target, cfg)
    char_count = len(_render_entries(entries))
    if limit > 0 and char_count > limit:
        raise MemoryRPCError("MEMORY_LIMIT_EXCEEDED", f"Memory would be {char_count}/{limit} chars.")


def _scan_entry(content: str) -> str | None:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Content contains invisible unicode character U+{ord(char):04X}."
    for pattern, risk_id in _THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Content matches blocked risk pattern '{risk_id}'."
    return None


def _preview_response(
    target: str,
    before_entries: list[str],
    after_entries: list[str],
    previous_hash: str,
    new_hash: str,
    cfg: dict[str, Any],
    *,
    config_available: bool,
) -> dict[str, Any]:
    before_status = _status_from_entries(target, before_entries, previous_hash, cfg, config_available=config_available)
    after_status = _status_from_entries(target, after_entries, new_hash, cfg, config_available=config_available)
    return {
        "target": target,
        "previous_revision_hash": previous_hash,
        "new_revision_hash": new_hash,
        "before": before_status,
        "after": after_status,
        "entry_diff": _entry_diff(before_entries, after_entries),
        "risk_flags": [],
        "session_warning": SESSION_WARNING,
    }


def _status_from_entries(
    target: str,
    entries: list[str],
    revision_hash: str,
    cfg: dict[str, Any],
    *,
    config_available: bool,
) -> dict[str, Any]:
    meta = TARGETS[target]
    limit = _char_limit(target, cfg)
    char_count = len(_render_entries(entries))
    return {
        "target": target,
        "label": meta["label"],
        "filename": meta["filename"],
        "enabled": bool(cfg.get(meta["enabled_key"], False)),
        "enabled_source": "config" if config_available else "unknown",
        "char_limit": limit,
        "char_count": char_count,
        "entry_count": len(entries),
        "capacity_state": _capacity_state(char_count, limit),
        "status": "empty" if not entries else ("over_capacity" if limit > 0 and char_count > limit else "readable"),
        "revision_hash": revision_hash,
        "mtime": None,
        "error": None,
    }


def _entry_diff(before_entries: list[str], after_entries: list[str]) -> dict[str, Any]:
    before_set = set(before_entries)
    after_set = set(after_entries)
    changed = sum(1 for idx, entry in enumerate(after_entries) if idx >= len(before_entries) or before_entries[idx] != entry)
    return {
        "added": len(after_set - before_set),
        "removed": len(before_set - after_set),
        "changed": changed,
        "before_count": len(before_entries),
        "after_count": len(after_entries),
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
        "revision_hash": _hash_text(""),
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


def _path_for_target(target: str) -> Path:
    if target not in TARGETS:
        raise MemoryRPCError("MEMORY_BAD_TARGET", "target must be 'memory' or 'user'.")
    return _memory_dir() / TARGETS[target]["filename"]


def _read_target_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MemoryRPCError("MEMORY_INVALID_FILE", f"Memory file is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise MemoryRPCError("MEMORY_UNREADABLE", f"Could not read memory file: {exc}") from exc


def _render_entries(entries: list[str]) -> str:
    return ENTRY_DELIMITER.join(entries) if entries else ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _char_limit(target: str, cfg: dict[str, Any]) -> int:
    meta = TARGETS[target]
    return _int_config(cfg.get(meta["limit_key"]), meta["default_limit"])


@contextmanager
def _file_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".vylen_mem_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _create_snapshot(target: str, path: Path, text: str, revision_hash: str, reason: str) -> str:
    snapshot_dir = _snapshot_dir(target)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = f"snap_{_compact_timestamp()}_{revision_hash[:8]}{os.urandom(2).hex()}"
    body_path = snapshot_dir / f"{snapshot_id}.md"
    meta_path = snapshot_dir / f"{snapshot_id}.json"
    _atomic_write_text(body_path, text)
    metadata = {
        "id": snapshot_id,
        "target": target,
        "label": TARGETS[target]["label"],
        "source_file": str(path),
        "revision_hash": revision_hash,
        "char_count": len(text),
        "entry_count": len(_parse_entries(text)),
        "reason": reason,
        "created_at": _now_iso(),
    }
    _atomic_write_text(meta_path, json.dumps(metadata, indent=2, sort_keys=True))
    return snapshot_id


def _snapshot_dir(target: str) -> Path:
    if target not in TARGETS:
        raise MemoryRPCError("MEMORY_BAD_TARGET", "target must be 'memory' or 'user'.")
    return _memory_dir() / ".vylen-snapshots" / target


def _valid_snapshot_id(snapshot_id: str) -> bool:
    return bool(re.fullmatch(r"snap_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}", snapshot_id))


def _snapshot_metadata_by_id(target: str, snapshot_id: str) -> dict[str, Any]:
    if not _valid_snapshot_id(snapshot_id):
        raise MemoryRPCError("MEMORY_BAD_SNAPSHOT", "snapshot_id is invalid.")
    meta_path = _snapshot_dir(target) / f"{snapshot_id}.json"
    metadata = _read_snapshot_metadata(meta_path)
    if metadata is None:
        raise MemoryRPCError("MEMORY_SNAPSHOT_NOT_FOUND", "Snapshot was not found.")
    body_path = _snapshot_dir(target) / f"{snapshot_id}.md"
    metadata["available"] = body_path.exists()
    if not metadata["available"]:
        raise MemoryRPCError("MEMORY_SNAPSHOT_NOT_FOUND", "Snapshot body was not found.")
    return metadata


def _read_snapshot_metadata(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("memory snapshot metadata unreadable: %s", path, exc_info=True)
        return None
    if not isinstance(raw, dict):
        return None
    snapshot_id = str(raw.get("id") or "")
    target = str(raw.get("target") or "")
    if target not in TARGETS or not _valid_snapshot_id(snapshot_id):
        return None
    return {
        "id": snapshot_id,
        "target": target,
        "label": str(raw.get("label") or TARGETS[target]["label"]),
        "revision_hash": str(raw.get("revision_hash") or ""),
        "char_count": _int_config(raw.get("char_count"), 0),
        "entry_count": _int_config(raw.get("entry_count"), 0),
        "reason": str(raw.get("reason") or ""),
        "created_at": str(raw.get("created_at") or ""),
    }


def _read_snapshot_body(target: str, snapshot_id: str) -> str:
    path = _snapshot_dir(target) / f"{snapshot_id}.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MemoryRPCError("MEMORY_SNAPSHOT_NOT_FOUND", "Snapshot body was not found.") from exc
    except UnicodeDecodeError as exc:
        raise MemoryRPCError("MEMORY_INVALID_FILE", f"Snapshot is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise MemoryRPCError("MEMORY_UNREADABLE", f"Could not read snapshot body: {exc}") from exc


def _compact_timestamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
