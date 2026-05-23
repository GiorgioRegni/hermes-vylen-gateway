"""Reads the two env vars that configure the gateway plugin.

Kept in its own module so the doctor CLI can use the same logic without
importing the adapter (which would in turn import Hermes if available).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

DEFAULT_CLOUD_URL = "https://relay.vylenagent.com"
GATEWAY_PATH = "/v1/gateway"


@dataclass(frozen=True)
class GatewayConfig:
    instance_token: str
    cloud_url: str  # the http(s) base, NOT the ws(s) URL
    websocket_url: str  # derived: ws(s)://host/v1/gateway

    @property
    def authorization_header(self) -> str:
        return f"Bearer {self.instance_token}"


class ConfigError(Exception):
    pass


def load_from_env() -> GatewayConfig:
    configs = load_all_from_env()
    return configs[0]


def load_all_from_env() -> list[GatewayConfig]:
    token = os.environ.get("VYLEN_INSTANCE_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "VYLEN_INSTANCE_TOKEN is not set. Get one from the Vylen Cloud "
            "portal (Add Hermes) and put it in ~/.hermes/.env."
        )
    raw_urls = os.environ.get("VYLEN_CLOUD_URLS", "").strip()
    if raw_urls:
        cloud_urls = _parse_cloud_urls(raw_urls, "VYLEN_CLOUD_URLS")
    else:
        cloud_urls = _parse_cloud_urls(
            os.environ.get("VYLEN_CLOUD_URL", DEFAULT_CLOUD_URL),
            "VYLEN_CLOUD_URL",
        )
    return [
        GatewayConfig(
            instance_token=token,
            cloud_url=cloud_url,
            websocket_url=_derive_ws_url(cloud_url),
        )
        for cloud_url in cloud_urls
    ]


def _parse_cloud_urls(raw: str, env_name: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        cloud_url = part.strip().rstrip("/")
        if not cloud_url:
            continue
        if cloud_url in seen:
            continue
        _derive_ws_url(cloud_url)
        urls.append(cloud_url)
        seen.add(cloud_url)
    if not urls:
        raise ConfigError(f"{env_name} does not contain a usable cloud URL")
    return urls


def _derive_ws_url(cloud_url: str) -> str:
    """Turn https://host -> wss://host/v1/gateway and http://host -> ws://host/v1/gateway."""
    parts = urlsplit(cloud_url)
    if parts.scheme not in ("http", "https"):
        raise ConfigError(f"Vylen cloud URLs must start with http:// or https://, got {cloud_url!r}")
    if not parts.netloc:
        raise ConfigError(f"Vylen cloud URL is missing a host: {cloud_url!r}")
    ws_scheme = "ws" if parts.scheme == "http" else "wss"
    return urlunsplit((ws_scheme, parts.netloc, GATEWAY_PATH, "", ""))
