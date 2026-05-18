# hermes-vylen-gateway

Vylen Cloud gateway plugin for [hermes-agent](https://github.com/NousResearch/hermes-agent).

The package connects a local Hermes process to Vylen Cloud over a single
outbound WebSocket authenticated by an instance token. From Vylen clients, the
local Hermes runtime appears as a paired instance with chat routing, cron
pushes, multimodal attachments, memory control-plane calls, and resumable
streaming all flowing through the same gateway session.

The plugin runs inside the Hermes process. It does not require Hermes'
loopback API server, a local API-server key, or inbound network access to the
machine running Hermes.

## Install

```bash
pip install hermes-vylen-gateway
hermes-vylen-gateway init
```

`hermes-vylen-gateway init` idempotently adds `vylen` to
`plugins.enabled` in Hermes' `config.yaml` so Hermes will load the package's
entry-point plugin.

Set the instance token in `~/.hermes/.env`, or in the environment where Hermes
runs:

```bash
VYLEN_INSTANCE_TOKEN=vyl_live_...   # one-time token from the Vylen portal
```

The plugin defaults to the production relay at
`https://relay.vylenagent.com`. Override only when pointing at another relay:

```bash
# Local dev against a Vylen Cloud running on your machine:
VYLEN_CLOUD_URL=http://localhost:8420

# From inside a Docker container where the host runs the cloud:
VYLEN_CLOUD_URL=http://host.docker.internal:8420
```

`http://` is accepted; the plugin selects `ws://` or `wss://` automatically.

Then start Hermes normally:

```bash
hermes gateway
```

## Verify Setup

```bash
vylen-gateway-doctor
```

The doctor performs the same dial, hello, and ready exchange the adapter uses
at runtime, then prints either `ok` or a diagnosable error. It is useful before
starting the full Hermes gateway.

Pairing codes can be exchanged from the CLI:

```bash
VYLEN_INSTANCE_TOKEN="$(vylen-gateway-pair ABC1-DEF2)"
```

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

Tests use in-process mock gateway components and do not require a Hermes
install unless a test explicitly says otherwise.

## Runtime Model

Hermes discovers this package through the `hermes_agent.plugins` entry-point
group and calls `hermes_vylen_gateway.register(ctx)`.

At runtime the adapter:

- opens one outbound WebSocket to Vylen Cloud;
- sends a token-authenticated `hello` frame and waits for `ready`;
- handles HTTP-shaped request frames in process through Hermes internals;
- streams response headers/chunks/end frames back to cloud;
- buffers active response streams locally for short-lived resume;
- emits `push` frames for Hermes cron output and generated media;
- serves short-lived local blob tokens through the existing gateway session.

See [docs/protocol.md](docs/protocol.md) for the public gateway contract and
[docs/hermes-internals.md](docs/hermes-internals.md) for Hermes plugin-loader
notes.

## Security

Vylen instance tokens and Hermes API/provider keys stay in the user's local
environment. The plugin does not store user message content in Vylen Cloud; any
short-lived response resume buffers live in the local plugin process.

Report vulnerabilities privately using the instructions in
[SECURITY.md](SECURITY.md).
