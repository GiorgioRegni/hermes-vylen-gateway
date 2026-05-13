"""`vylen-gateway-pair` — exchange a portal-issued pairing code for an
instance token. Mirrors what BotFather → user does on Telegram, just in CLI
form.

Default output is just the token on stdout, so the natural setup pattern is:

    echo "VYLEN_INSTANCE_TOKEN=$(vylen-gateway-pair ABC1-DEF2)" >> ~/.hermes/.env

With `--verbose`, the command prints a multi-line summary instead — useful
when the user is running it interactively rather than from a shell snippet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from .config import DEFAULT_CLOUD_URL


class PairError(Exception):
    pass


def exchange(cloud_url: str, code: str, *, timeout: float = 10.0) -> tuple[str, str]:
    """Exchange `code` for an instance token. Returns (instance_id, token)."""
    payload = json.dumps({"pairing_code": code}).encode("utf-8")
    req = urllib.request.Request(
        cloud_url.rstrip("/") + "/v1/instances/pair",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read().decode("utf-8"))
            msg = err.get("error", {}).get("message") or str(exc)
        except Exception:  # noqa: BLE001
            msg = str(exc)
        raise PairError(f"cloud rejected pairing code: {msg}") from exc
    except urllib.error.URLError as exc:
        raise PairError(f"could not reach {cloud_url}: {exc.reason}") from exc

    instance_id = body.get("instance_id") or ""
    token = body.get("instance_token") or ""
    if not instance_id or not token:
        raise PairError(f"unexpected response shape: {body!r}")
    return instance_id, token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vylen-gateway-pair",
        description=(
            "Exchange a pairing code (from the Vylen Cloud portal) for an "
            "instance token. Default output is just the token on stdout so "
            "the result drops straight into VYLEN_INSTANCE_TOKEN=$(…)."
        ),
    )
    parser.add_argument("code", help='Pairing code, e.g. "ABC1-DEF2".')
    parser.add_argument(
        "--cloud-url",
        default=os.environ.get("VYLEN_CLOUD_URL", DEFAULT_CLOUD_URL),
        help="Override the cloud URL (else $VYLEN_CLOUD_URL or production default).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print a human-readable summary instead of just the token.",
    )
    args = parser.parse_args(argv)

    try:
        instance_id, token = exchange(args.cloud_url, args.code)
    except PairError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.verbose:
        print(f"cloud      : {args.cloud_url}")
        print(f"instance_id: {instance_id}")
        print(f"token      : {token}")
        print()
        print("Next: put this in ~/.hermes/.env and run `hermes gateway`:")
        print(f"  VYLEN_INSTANCE_TOKEN={token}")
        if args.cloud_url != DEFAULT_CLOUD_URL:
            print(f"  VYLEN_CLOUD_URL={args.cloud_url}")
    else:
        print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
