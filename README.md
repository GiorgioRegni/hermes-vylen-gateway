# hermes-vylen-gateway

Vylen Cloud gateway plugin for [hermes-agent](https://github.com/NousResearch/hermes-agent).

Connects a local Hermes process to Vylen Cloud over a single outbound
WebSocket, authenticated by an instance token. From Vylen's mobile/web
clients, this Hermes appears as a registered instance — message routing,
cron push, and multimodal flow over the same socket.

Implemented per
[`docs/specs/001-hermes-gateway.md`](../docs/specs/001-hermes-gateway.md)
in the Vylen monorepo.

## Setup

```bash
pip install hermes-vylen-gateway
```

Set the instance token in `~/.hermes/.env` (or wherever you run Hermes from):

```bash
VYLEN_INSTANCE_TOKEN=vyl_live_…   # one-time, from the Vylen portal
```

The plugin defaults to the production relay at `https://relay.vylenagent.com`. Override only when pointing at something else:

```bash
# Local dev against a Vylen Cloud running on your machine:
VYLEN_CLOUD_URL=http://localhost:8420

# Or from inside a Docker container where the host runs the cloud:
VYLEN_CLOUD_URL=http://host.docker.internal:8420
```

`http://` is accepted; the plugin selects `ws://` vs `wss://` automatically.

Then start Hermes the normal way:

```bash
hermes gateway
```

Hermes discovers this plugin through the `hermes_agent.plugins` entry point
and registers the `vylen` platform automatically.

## Verifying setup

```bash
vylen-gateway-doctor
```

Performs the same dial + hello + ready exchange the adapter does, prints
either `ok` or a diagnosable error. Useful before you spin up the full
`hermes gateway`.

## Dev / testing

```bash
pip install -e '.[dev]'
pytest
```

Tests stand up a tiny in-process `websockets` server as the mock cloud — no
Hermes install needed.

## How Hermes loads this plugin

See [docs/hermes-internals.md](docs/hermes-internals.md). Entry-point
contract, `plugins.enabled` gate, Docker quirks, and debugging recipes are
all there. Read it once before touching the adapter or the entry point.
