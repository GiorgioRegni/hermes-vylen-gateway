from __future__ import annotations

import os

import hermes_vylen_gateway


class FakeContext:
    def __init__(self) -> None:
        self.platforms: list[dict] = []

    def register_platform(self, **kwargs) -> None:
        self.platforms.append(kwargs)


def test_register_declares_vylen_scoped_auth_envs(monkeypatch):
    monkeypatch.delenv("VYLEN_HOME_CHAT_ID", raising=False)
    ctx = FakeContext()

    hermes_vylen_gateway.register(ctx)

    assert ctx.platforms
    platform = ctx.platforms[0]
    assert platform["name"] == "vylen"
    assert platform["allowed_users_env"] == "VYLEN_ALLOWED_USERS"
    assert platform["allow_all_env"] == "VYLEN_ALLOW_ALL_USERS"
    assert platform["cron_deliver_env_var"] == "VYLEN_HOME_CHAT_ID"
    assert os.environ["VYLEN_HOME_CHAT_ID"] == "inbox"
