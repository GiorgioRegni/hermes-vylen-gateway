"""hermes-vylen-gateway — entry point for Hermes's plugin loader.

`hermes_cli/plugins.py` discovers us through the `hermes_agent.plugins`
entry-point group declared in pyproject.toml and calls `register(ctx)` with a
context object that exposes `register_platform(...)`.
"""

from __future__ import annotations

import os

from .adapter import adapter_factory, check_dependencies
from .client import HandshakeError, ReadyInfo, VylenGatewayClient
from .config import GatewayConfig, ConfigError, load_from_env

__all__ = [
    "register",
    "GatewayConfig",
    "ConfigError",
    "HandshakeError",
    "ReadyInfo",
    "VylenGatewayClient",
    "load_from_env",
]


def register(ctx) -> None:
    """Called by Hermes's plugin loader on `hermes gateway` startup."""
    # Hermes's cron resolver gates `deliver=<platform>` on (a) the platform
    # declaring a `cron_deliver_env_var` and (b) that env var holding a
    # non-empty "home chat id" at resolution time. The chat_id is just a
    # routing bucket — Vylen fans out by user_id, not by chat_id, so the
    # value is essentially decorative. Default it to "inbox" so the user
    # doesn't have to set it manually; allow override by setting
    # VYLEN_HOME_CHAT_ID explicitly (useful if a future revision splits the
    # inbox into multiple buckets by chat_id).
    os.environ.setdefault("VYLEN_HOME_CHAT_ID", "inbox")
    ctx.register_platform(
        name="vylen",
        label="Vylen",
        adapter_factory=adapter_factory,
        check_fn=check_dependencies,
        required_env=["VYLEN_INSTANCE_TOKEN"],
        install_hint=(
            "Add VYLEN_INSTANCE_TOKEN to ~/.hermes/.env "
            "(get a token from the Vylen Cloud portal)."
        ),
        emoji="🚀",
        cron_deliver_env_var="VYLEN_HOME_CHAT_ID",
    )
