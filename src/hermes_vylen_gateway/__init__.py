"""hermes-vylen-gateway — entry point for Hermes's plugin loader.

`hermes_cli/plugins.py` discovers us through the `hermes_agent.plugins`
entry-point group declared in pyproject.toml and calls `register(ctx)` with a
context object that exposes `register_platform(...)`.
"""

from __future__ import annotations

import logging
import os

from .adapter import VYLEN_INBOX_CHAT_ID, adapter_factory, check_dependencies
from .client import HandshakeError, ReadyInfo, VylenGatewayClient
from .config import GatewayConfig, ConfigError, load_all_from_env, load_from_env

__all__ = [
    "register",
    "GatewayConfig",
    "ConfigError",
    "HandshakeError",
    "ReadyInfo",
    "VylenGatewayClient",
    "load_all_from_env",
    "load_from_env",
]


def _vylen_env_enablement() -> dict:
    """Seed config.platforms[vylen] when the platform auto-enables.

    Declares the Vylen home channel so the agent's send_message(target="vylen")
    resolves to the notifications inbox bucket instead of erroring. The chat_id
    is the same synthetic "inbox" bucket cron delivery and all plugin-initiated
    pushes already use (Vylen fans out by user_id, not chat_id).
    """
    chat_id = (os.environ.get("VYLEN_HOME_CHAT_ID") or VYLEN_INBOX_CHAT_ID).strip() \
        or VYLEN_INBOX_CHAT_ID
    return {"home_channel": {"chat_id": chat_id, "name": "Vylen"}}


def register(ctx) -> None:
    """Called by Hermes's plugin loader on `hermes gateway` startup."""
    # Surface the supervisor's lifecycle events (gateway online, socket
    # dropped, reconnect-in-Ns) in the Hermes container log. Hermes
    # configures the root logger but third-party loggers default to
    # WARNING — without this explicit override our INFO logs are
    # filtered out, which made past "is the WS alive?" debugging mean
    # tailing for absence-of-evidence rather than positive proof.
    #
    # Override the verbosity via VYLEN_LOG_LEVEL=DEBUG|INFO|WARNING|...
    # if the default chatter is too much (it's low-volume; every 15s
    # health probe is at DEBUG, only reconnects and drops are INFO).
    _configure_plugin_logger()
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
        allowed_users_env="VYLEN_ALLOWED_USERS",
        allow_all_env="VYLEN_ALLOW_ALL_USERS",
        cron_deliver_env_var="VYLEN_HOME_CHAT_ID",
        # Makes send_message(target="vylen") resolve the home channel to the
        # notifications inbox bucket; reuses VYLEN_HOME_CHAT_ID so cron and
        # send_message agree on the bucket.
        env_enablement_fn=_vylen_env_enablement,
    )


_LOG_HANDLER_TAG = "_vylen_plugin_handler"


def _configure_plugin_logger() -> None:
    """Apply VYLEN_LOG_LEVEL (default INFO) to the plugin's package
    logger and attach a dedicated stderr handler so our INFO records
    aren't filtered out by Hermes's root logger (which is configured at
    WARNING by gateway.run). Idempotent — safe to call from every
    process that re-imports the plugin (CLI, gateway, cron scheduler).
    """
    raw = (os.environ.get("VYLEN_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        level = logging.INFO
    package_logger = logging.getLogger("hermes_vylen_gateway")
    package_logger.setLevel(level)
    # Stop the record from also being handed to the root logger, which
    # otherwise applies its own (typically WARNING) filter and swallows
    # our INFO output. We own the handler below; root's handler doesn't
    # get a second chance.
    package_logger.propagate = False
    # Attach our handler once per process. Tag it so a re-import doesn't
    # stack duplicates if Hermes calls `discover_plugins()` more than
    # once in the same interpreter.
    for existing in package_logger.handlers:
        if getattr(existing, _LOG_HANDLER_TAG, False):
            existing.setLevel(level)
            return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    setattr(handler, _LOG_HANDLER_TAG, True)
    package_logger.addHandler(handler)
