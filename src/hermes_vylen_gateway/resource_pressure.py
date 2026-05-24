from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

CPU_WARNING_ENTER = 75.0
CPU_WARNING_EXIT = 65.0
CPU_CRITICAL_ENTER = 90.0
CPU_CRITICAL_EXIT = 80.0

MEM_WARNING_RATIO_ENTER = 0.15
MEM_WARNING_RATIO_EXIT = 0.20
MEM_CRITICAL_RATIO_ENTER = 0.08
MEM_CRITICAL_RATIO_EXIT = 0.12
MEM_WARNING_BYTES_ENTER = int(1.5 * 1024 * 1024 * 1024)
MEM_WARNING_BYTES_EXIT = 2 * 1024 * 1024 * 1024
MEM_CRITICAL_BYTES_ENTER = 512 * 1024 * 1024
MEM_CRITICAL_BYTES_EXIT = 1024 * 1024 * 1024


@dataclass(frozen=True)
class ResourceMetrics:
    cpu_percent: float | None = None
    memory_available_ratio: float | None = None
    memory_available_bytes: int | None = None


class ResourceMetricsProvider(Protocol):
    def sample(self) -> ResourceMetrics: ...


class ResourcePressureSampler:
    def __init__(
        self,
        provider: ResourceMetricsProvider,
        now: Callable[[], float] | None = None,
    ):
        self._provider = provider
        self._now = now
        self._cpu = _CPUState()
        self._memory = _MemoryState()

    def sample(self) -> dict[str, str]:
        try:
            metrics = self._provider.sample()
        except Exception:  # noqa: BLE001
            metrics = ResourceMetrics()
        now = float(self._now()) if self._now is not None else time.monotonic()
        return {
            "cpu": self._cpu.update(metrics.cpu_percent, now),
            "memory": self._memory.update(
                metrics.memory_available_ratio,
                metrics.memory_available_bytes,
                now,
            ),
        }


class _BucketState:
    def __init__(self) -> None:
        self.bucket = "unknown"
        self._candidate: str | None = None
        self._candidate_since = 0.0

    def _transition(self, candidate: str | None, dwell_s: float, now: float) -> str:
        if candidate is None or candidate == self.bucket:
            self._candidate = None
            self._candidate_since = 0.0
            return self.bucket
        if self._candidate != candidate:
            self._candidate = candidate
            self._candidate_since = now
            return self.bucket
        if now - self._candidate_since >= dwell_s:
            self.bucket = candidate
            self._candidate = None
            self._candidate_since = 0.0
        return self.bucket

    def _set_immediate(self, bucket: str) -> str:
        self.bucket = bucket
        self._candidate = None
        self._candidate_since = 0.0
        return self.bucket


class _CPUState(_BucketState):
    def update(self, value: float | None, now: float) -> str:
        if value is None:
            return self._set_immediate("unknown")
        if self.bucket == "unknown":
            self.bucket = "ok"
        if self.bucket == "critical":
            if value < CPU_CRITICAL_EXIT:
                return self._transition(
                    "ok" if value < CPU_WARNING_EXIT else "warning",
                    60.0,
                    now,
                )
            return self._transition(None, 0.0, now)
        if self.bucket == "warning":
            if value >= CPU_CRITICAL_ENTER:
                return self._transition("critical", 15.0, now)
            if value < CPU_WARNING_EXIT:
                return self._transition("ok", 60.0, now)
            return self._transition(None, 0.0, now)
        if value >= CPU_CRITICAL_ENTER:
            return self._transition("critical", 15.0, now)
        if value >= CPU_WARNING_ENTER:
            return self._transition("warning", 30.0, now)
        return self._transition(None, 0.0, now)


class _MemoryState(_BucketState):
    def update(self, ratio: float | None, available_bytes: int | None, now: float) -> str:
        if ratio is None or available_bytes is None:
            return self._set_immediate("unknown")
        if self.bucket == "unknown":
            self.bucket = "ok"
        critical_enter = (
            ratio <= MEM_CRITICAL_RATIO_ENTER
            or available_bytes <= MEM_CRITICAL_BYTES_ENTER
        )
        warning_enter = (
            ratio <= MEM_WARNING_RATIO_ENTER
            or available_bytes <= MEM_WARNING_BYTES_ENTER
        )
        critical_exit = (
            ratio >= MEM_CRITICAL_RATIO_EXIT
            and available_bytes >= MEM_CRITICAL_BYTES_EXIT
        )
        warning_exit = (
            ratio >= MEM_WARNING_RATIO_EXIT
            and available_bytes >= MEM_WARNING_BYTES_EXIT
        )
        if self.bucket == "critical":
            if critical_exit:
                return self._transition("ok" if warning_exit else "warning", 60.0, now)
            return self._transition(None, 0.0, now)
        if self.bucket == "warning":
            if critical_enter:
                return self._transition("critical", 15.0, now)
            if warning_exit:
                return self._transition("ok", 60.0, now)
            return self._transition(None, 0.0, now)
        if critical_enter:
            return self._transition("critical", 15.0, now)
        if warning_enter:
            return self._transition("warning", 30.0, now)
        return self._transition(None, 0.0, now)


class PsutilMetricsProvider:
    def __init__(self, psutil_module: object):
        self._psutil = psutil_module

    def sample(self) -> ResourceMetrics:
        cpu = self._psutil.cpu_percent(interval=None)
        mem = self._psutil.virtual_memory()
        total = float(getattr(mem, "total", 0) or 0)
        available = int(getattr(mem, "available", 0) or 0)
        ratio = float(available) / total if total > 0 else None
        return ResourceMetrics(
            cpu_percent=float(cpu),
            memory_available_ratio=ratio,
            memory_available_bytes=available,
        )


class StdlibMetricsProvider:
    def sample(self) -> ResourceMetrics:
        meminfo = _read_meminfo()
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        ratio = float(available) / float(total) if total and available is not None else None
        return ResourceMetrics(
            cpu_percent=_loadavg_cpu_percent(),
            memory_available_ratio=ratio,
            memory_available_bytes=available,
        )


def build_resource_pressure_sampler_from_env() -> ResourcePressureSampler | None:
    if not _enabled(os.environ.get("VYLEN_RESOURCE_PRESSURE_ENABLED")):
        return None
    return ResourcePressureSampler(_best_provider())


def _enabled(raw: str | None) -> bool:
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _best_provider() -> ResourceMetricsProvider:
    try:
        return PsutilMetricsProvider(importlib.import_module("psutil"))
    except Exception:  # noqa: BLE001
        return StdlibMetricsProvider()


def _loadavg_cpu_percent() -> float | None:
    try:
        load1 = os.getloadavg()[0]
        cpus = os.cpu_count() or 0
    except (AttributeError, OSError):
        return None
    if cpus <= 0:
        return None
    return min(100.0, max(0.0, (load1 / cpus) * 100.0))


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                raw_value = rest.strip().split()[0] if rest.strip() else ""
                if raw_value.isdigit():
                    values[name] = int(raw_value) * 1024
    except OSError:
        return {}
    return values
