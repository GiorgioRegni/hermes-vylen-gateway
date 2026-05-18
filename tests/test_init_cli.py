from __future__ import annotations

import ast

from hermes_vylen_gateway import init_cli
from hermes_vylen_gateway.init_cli import enable_plugin, main


class FakeYaml:
    @staticmethod
    def safe_load(raw: str):
        return ast.literal_eval(raw) if raw.strip() else None

    @staticmethod
    def safe_dump(data, sort_keys=False):
        return repr(data)


def test_enable_plugin_creates_missing_config(tmp_path):
    init_cli._load_yaml = lambda: FakeYaml
    config = tmp_path / "config.yaml"

    changed, path = enable_plugin(config)

    assert changed is True
    assert path == config
    assert FakeYaml.safe_load(config.read_text()) == {"plugins": {"enabled": ["vylen"]}}


def test_enable_plugin_is_idempotent_and_preserves_config(tmp_path):
    init_cli._load_yaml = lambda: FakeYaml
    config = tmp_path / "config.yaml"
    config.write_text(
        FakeYaml.safe_dump({
            "profile": "dev",
            "plugins": {"enabled": ["other", "vylen"]},
            "platforms": {"api_server": {"enabled": True}},
        })
    )

    changed, _ = enable_plugin(config)
    data = FakeYaml.safe_load(config.read_text())

    assert changed is False
    assert data["plugins"]["enabled"] == ["other", "vylen"]
    assert data["platforms"]["api_server"]["enabled"] is True


def test_init_cli_uses_explicit_config(tmp_path, capsys):
    init_cli._load_yaml = lambda: FakeYaml
    config = tmp_path / "config.yaml"

    rc = main(["init", "--config", str(config)])

    assert rc == 0
    assert "enabled vylen" in capsys.readouterr().out
    assert "api_server" not in config.read_text()
