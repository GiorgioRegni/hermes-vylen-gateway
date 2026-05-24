from __future__ import annotations

from hermes_vylen_gateway.resource_pressure import (
    ResourceMetrics,
    ResourcePressureSampler,
    _enabled,
)


class FakeProvider:
    def __init__(self):
        self.metrics = ResourceMetrics()

    def sample(self) -> ResourceMetrics:
        return self.metrics


def test_cpu_warning_and_exit_hysteresis():
    provider = FakeProvider()
    now = 0.0
    sampler = ResourcePressureSampler(provider, now=lambda: now)

    provider.metrics = ResourceMetrics(
        cpu_percent=80.0,
        memory_available_ratio=0.50,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
    )
    assert sampler.sample()["cpu"] == "ok"
    now = 29.0
    assert sampler.sample()["cpu"] == "ok"
    now = 30.0
    assert sampler.sample()["cpu"] == "warning"

    provider.metrics = ResourceMetrics(
        cpu_percent=60.0,
        memory_available_ratio=0.50,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
    )
    now = 89.0
    assert sampler.sample()["cpu"] == "warning"
    now = 149.0
    assert sampler.sample()["cpu"] == "ok"


def test_memory_critical_and_exit_hysteresis():
    provider = FakeProvider()
    now = 0.0
    sampler = ResourcePressureSampler(provider, now=lambda: now)

    provider.metrics = ResourceMetrics(
        cpu_percent=5.0,
        memory_available_ratio=0.07,
        memory_available_bytes=2 * 1024 * 1024 * 1024,
    )
    assert sampler.sample()["memory"] == "ok"
    now = 14.0
    assert sampler.sample()["memory"] == "ok"
    now = 15.0
    assert sampler.sample()["memory"] == "critical"

    provider.metrics = ResourceMetrics(
        cpu_percent=5.0,
        memory_available_ratio=0.25,
        memory_available_bytes=3 * 1024 * 1024 * 1024,
    )
    now = 74.0
    assert sampler.sample()["memory"] == "critical"
    now = 134.0
    assert sampler.sample()["memory"] == "ok"


def test_metric_unavailable_returns_unknown_and_resets_candidate():
    provider = FakeProvider()
    now = 0.0
    sampler = ResourcePressureSampler(provider, now=lambda: now)

    provider.metrics = ResourceMetrics(
        cpu_percent=95.0,
        memory_available_ratio=0.50,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
    )
    assert sampler.sample()["cpu"] == "ok"
    now = 10.0
    provider.metrics = ResourceMetrics()
    assert sampler.sample() == {"cpu": "unknown", "memory": "unknown"}

    now = 20.0
    provider.metrics = ResourceMetrics(
        cpu_percent=95.0,
        memory_available_ratio=0.50,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
    )
    assert sampler.sample()["cpu"] == "ok"


def test_critical_recovery_steps_down_to_warning_before_ok():
    provider = FakeProvider()
    now = 0.0
    sampler = ResourcePressureSampler(provider, now=lambda: now)

    provider.metrics = ResourceMetrics(
        cpu_percent=95.0,
        memory_available_ratio=0.07,
        memory_available_bytes=400 * 1024 * 1024,
    )
    assert sampler.sample() == {"cpu": "ok", "memory": "ok"}
    now = 15.0
    assert sampler.sample() == {"cpu": "critical", "memory": "critical"}

    provider.metrics = ResourceMetrics(
        cpu_percent=70.0,
        memory_available_ratio=0.18,
        memory_available_bytes=int(1.6 * 1024 * 1024 * 1024),
    )
    now = 75.0
    assert sampler.sample() == {"cpu": "critical", "memory": "critical"}
    now = 135.0
    assert sampler.sample() == {"cpu": "warning", "memory": "warning"}


def test_enabled_env_parser_treats_zero_as_disabled():
    assert _enabled(None)
    assert _enabled("")
    assert _enabled("1")
    assert not _enabled("0")
    assert not _enabled("false")
    assert not _enabled("off")
