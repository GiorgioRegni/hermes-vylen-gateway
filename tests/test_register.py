from __future__ import annotations

import os
import types

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


def test_register_declares_home_channel_env_enablement(monkeypatch):
    monkeypatch.delenv("VYLEN_HOME_CHAT_ID", raising=False)
    ctx = FakeContext()

    hermes_vylen_gateway.register(ctx)

    platform = ctx.platforms[0]
    env_enablement_fn = platform["env_enablement_fn"]
    assert callable(env_enablement_fn)
    assert env_enablement_fn() == {
        "home_channel": {"chat_id": "inbox", "name": "Vylen"}
    }


def test_home_channel_follows_vylen_home_chat_id(monkeypatch):
    monkeypatch.setenv("VYLEN_HOME_CHAT_ID", "custombucket")
    ctx = FakeContext()

    hermes_vylen_gateway.register(ctx)

    env_enablement_fn = ctx.platforms[0]["env_enablement_fn"]
    assert env_enablement_fn()["home_channel"]["chat_id"] == "custombucket"


def test_home_channel_satisfies_hermes_extraction(monkeypatch):
    """Replicates gateway/config.py:1909-1921 home_channel extraction.

    Kept hermetic (no hermes-agent import): asserts the env_enablement_fn
    return value carries exactly what Hermes's extraction consumes.
    """
    monkeypatch.delenv("VYLEN_HOME_CHAT_ID", raising=False)
    ctx = FakeContext()
    hermes_vylen_gateway.register(ctx)

    seed = dict(ctx.platforms[0]["env_enablement_fn"]())
    home = seed.pop("home_channel")
    chat_id = home.get("chat_id")
    assert chat_id
    name = home.get("name") or "Home"

    home_channel = types.SimpleNamespace(chat_id=chat_id, name=name)
    assert home_channel.chat_id == "inbox"
    assert home_channel.name == "Vylen"
