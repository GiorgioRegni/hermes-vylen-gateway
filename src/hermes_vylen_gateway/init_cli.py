"""Installer helper for enabling the Vylen Hermes plugin."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-vylen-gateway",
        description="Manage the Vylen Hermes gateway plugin.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Enable the vylen plugin in Hermes config.yaml.")
    init.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Hermes config.yaml. Defaults to $HERMES_HOME/config.yaml or ~/.hermes/config.yaml.",
    )
    args = parser.parse_args(argv)
    if args.command == "init":
        try:
            changed, config_path = enable_plugin(args.config)
        except InitError as exc:
            print(f"init error: {exc}", file=sys.stderr)
            return 1
        if changed:
            print(f"enabled vylen in {config_path}")
        else:
            print(f"vylen already enabled in {config_path}")
        return 0
    parser.print_help(sys.stderr)
    return 2


class InitError(Exception):
    pass


def enable_plugin(config_path: Path | None = None) -> tuple[bool, Path]:
    yaml = _load_yaml()
    path = config_path or _default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise InitError(f"could not parse {path}: {exc}") from exc
    else:
        cfg = {}
    if not isinstance(cfg, dict):
        raise InitError(f"{path} must contain a YAML object at the top level")

    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        raise InitError(f"{path}: plugins must be a YAML object")
    enabled = plugins.get("enabled")
    if enabled is None:
        enabled = []
    if not isinstance(enabled, list):
        raise InitError(f"{path}: plugins.enabled must be a YAML list")
    if "vylen" in enabled:
        return False, path

    enabled.append("vylen")
    plugins["enabled"] = enabled
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return True, path


def _default_config_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "config.yaml"
    return Path.home() / ".hermes" / "config.yaml"


def _load_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise InitError(
            "PyYAML is not available in this Hermes environment; install Hermes with YAML support "
            "or edit config.yaml manually to add plugins.enabled: [vylen]."
        ) from exc
    return yaml


if __name__ == "__main__":
    sys.exit(main())
