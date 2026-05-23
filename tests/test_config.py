from __future__ import annotations

import pytest

from hermes_vylen_gateway import adapter
from hermes_vylen_gateway.config import ConfigError, GatewayConfig, load_all_from_env, load_from_env


def test_load_all_from_env_parses_cloud_url_list(monkeypatch):
    monkeypatch.setenv("VYLEN_INSTANCE_TOKEN", "vyl_live_test")
    monkeypatch.setenv(
        "VYLEN_CLOUD_URLS",
        " https://relay-a.example.test/ , http://127.0.0.1:8420, https://relay-a.example.test ",
    )
    monkeypatch.delenv("VYLEN_CLOUD_URL", raising=False)

    configs = load_all_from_env()

    assert [cfg.cloud_url for cfg in configs] == [
        "https://relay-a.example.test",
        "http://127.0.0.1:8420",
    ]
    assert [cfg.websocket_url for cfg in configs] == [
        "wss://relay-a.example.test/v1/gateway",
        "ws://127.0.0.1:8420/v1/gateway",
    ]


def test_load_from_env_falls_back_to_single_cloud_url(monkeypatch):
    monkeypatch.setenv("VYLEN_INSTANCE_TOKEN", "vyl_live_test")
    monkeypatch.delenv("VYLEN_CLOUD_URLS", raising=False)
    monkeypatch.setenv("VYLEN_CLOUD_URL", "http://localhost:8420/")

    config = load_from_env()

    assert config.cloud_url == "http://localhost:8420"
    assert config.websocket_url == "ws://localhost:8420/v1/gateway"


def test_load_all_from_env_rejects_invalid_cloud_urls(monkeypatch):
    monkeypatch.setenv("VYLEN_INSTANCE_TOKEN", "vyl_live_test")
    monkeypatch.setenv("VYLEN_CLOUD_URLS", "relay.example.test")

    with pytest.raises(ConfigError):
        load_all_from_env()


@pytest.mark.asyncio
async def test_rank_gateway_configs_orders_by_latency_and_keeps_down_relays_last(monkeypatch):
    configs = [
        GatewayConfig("token", "https://slow.example.test", "wss://slow.example.test/v1/gateway"),
        GatewayConfig("token", "https://down.example.test", "wss://down.example.test/v1/gateway"),
        GatewayConfig("token", "https://fast.example.test", "wss://fast.example.test/v1/gateway"),
    ]

    async def fake_latency(config: GatewayConfig) -> float | None:
        return {
            "https://slow.example.test": 0.2,
            "https://down.example.test": None,
            "https://fast.example.test": 0.05,
        }[config.cloud_url]

    monkeypatch.setattr(adapter, "_relay_latency", fake_latency)

    ranked = await adapter._rank_gateway_configs(configs)

    assert [cfg.cloud_url for cfg in ranked] == [
        "https://fast.example.test",
        "https://slow.example.test",
        "https://down.example.test",
    ]
