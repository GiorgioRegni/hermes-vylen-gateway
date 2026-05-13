"""`vylen-gateway-doctor` — verifies that the local env can complete the
gateway handshake against the configured cloud, without requiring Hermes to
be installed. Intended for users to debug setup before running
`hermes gateway`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .client import HandshakeError, VylenGatewayClient
from .config import ConfigError, load_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vylen-gateway-doctor",
        description=(
            "Verify that VYLEN_INSTANCE_TOKEN + VYLEN_CLOUD_URL can complete a "
            "hello/ready handshake against the configured Vylen Cloud."
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Handshake timeout in seconds."
    )
    parser.add_argument(
        "--keep-open",
        type=float,
        default=0.0,
        help="Hold the socket open for this many seconds after the handshake "
        "(useful for verifying instance shows 'connected' in the portal).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(f"cloud  : {cfg.cloud_url}")
    print(f"wsurl  : {cfg.websocket_url}")
    print(f"token  : {cfg.instance_token[:14]}…")

    try:
        ready = asyncio.run(_run(cfg, args.timeout, args.keep_open))
    except HandshakeError as exc:
        print(f"handshake failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    print(f"ready  : instance_id={ready.instance_id} user_id={ready.user_id}")
    print("ok")
    return 0


async def _run(cfg, timeout: float, keep_open: float):
    client = VylenGatewayClient(cfg)
    try:
        ready = await client.connect(timeout=timeout)
        if keep_open > 0:
            print(f"holding socket for {keep_open:.1f}s …")
            await asyncio.sleep(keep_open)
        return ready
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(main())
