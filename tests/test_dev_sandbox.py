from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dev_firebase_preserves_persistent_container_pairing():
    makefile_path = REPO_ROOT / "Makefile"
    if not makefile_path.exists():
        pytest.skip("private monorepo Makefile is not present in the public package repo")
    makefile = makefile_path.read_text(encoding="utf-8")
    dev_firebase = makefile.split("\ndev-firebase:\n", 1)[1].split("\n\ndev-cloud-firebase:", 1)[0]
    target = makefile.split("\ndev-hermes-up:\n", 1)[1].split("\n\ndev-hermes-pair:", 1)[0]

    assert "$(MAKE) --no-print-directory dev-hermes-up HERMES_DEV_PRESERVE_PAIRING=1" in dev_firebase
    assert 'dev-hermes-pair HERMES_DEV_PAIRING_CODE="$(HERMES_DEV_PAIRING_CODE)"' in makefile
    assert 'dev-hermes-up HERMES_DEV_PRESERVE_PAIRING=1' in makefile
    assert 'HERMES_DEV_PRESERVE_PAIRING ?= 0' in makefile
    assert 'grep -q "^VYLEN_INSTANCE_TOKEN=" "$$env_file"' in target
    assert 'grep -v "^VYLEN_CLOUD_URL=" "$$env_file"' in target
    assert 'printf "VYLEN_INSTANCE_TOKEN=vyl_dev_local_v1\\nVYLEN_CLOUD_URL=$(HERMES_DEV_CLOUD_URL)\\n"' in target
    assert "hermes-vylen-gateway init --config /opt/data/config.yaml" in target
    assert target.index("VYLEN_INSTANCE_TOKEN=vyl_dev_local_v1") < target.index("hermes-vylen-gateway init --config")
    assert target.index("hermes-vylen-gateway init --config") < target.index("restart hermes-dev")


def test_dev_hermes_compose_does_not_enable_global_allow_all_users():
    compose_path = REPO_ROOT / "dev" / "hermes-compose.yml"
    if not compose_path.exists():
        pytest.skip("private monorepo dev compose file is not present in the public package repo")
    compose = compose_path.read_text(encoding="utf-8")

    assert "GATEWAY_ALLOW_ALL_USERS" not in compose
    assert "VYLEN_ALLOW_ALL_USERS" not in compose
    assert 'VYLEN_INSTANCE_TOKEN: "vyl_dev_local_v1"' in compose
