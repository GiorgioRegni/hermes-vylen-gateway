"""hermes-vylen-gateway — entry point for Hermes's plugin loader.

`hermes_cli/plugins.py` discovers us through the `hermes_agent.plugins`
entry-point group declared in pyproject.toml and calls `register(ctx)` with a
context object that exposes `register_platform(...)`.
"""

from __future__ import annotations

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
    )
