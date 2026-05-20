from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dev_hermes_up_rewrites_persistent_container_env_before_restart():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("\ndev-hermes-up:\n", 1)[1].split("\n\ndev-hermes-pair:", 1)[0]

    assert 'printf "VYLEN_INSTANCE_TOKEN=vyl_dev_local_v1\\nVYLEN_CLOUD_URL=$(HERMES_DEV_CLOUD_URL)\\n"' in target
    assert "> /opt/data/.env" in target
    assert "hermes-vylen-gateway init --config /opt/data/config.yaml" in target
    assert target.index("> /opt/data/.env") < target.index("hermes-vylen-gateway init --config")
    assert target.index("hermes-vylen-gateway init --config") < target.index("restart hermes-dev")


def test_dev_hermes_compose_does_not_enable_global_allow_all_users():
    compose = (REPO_ROOT / "dev" / "hermes-compose.yml").read_text(encoding="utf-8")

    assert "GATEWAY_ALLOW_ALL_USERS" not in compose
    assert "VYLEN_ALLOW_ALL_USERS" not in compose
    assert 'VYLEN_INSTANCE_TOKEN: "vyl_dev_local_v1"' in compose
