# Hermes Integration Internals

This document records the Hermes-specific loader and runtime behavior that the
Vylen gateway plugin depends on. It is intentionally focused on the public
plugin package, not on Vylen Cloud internals.

## How Hermes Discovers Plugins

Hermes scans plugins from three sources, in priority order:

1. Bundled plugins under the Hermes repository.
2. User plugins under `~/.hermes/plugins/<name>/`.
3. Python entry-point packages that declare
   `[project.entry-points."hermes_agent.plugins"]`.

This package uses the third mechanism:

```toml
[project.entry-points."hermes_agent.plugins"]
vylen = "hermes_vylen_gateway"
```

Hermes loads the module and then calls its top-level `register(ctx)` function.
The entry point must point at the module, not at `hermes_vylen_gateway:register`.

## `plugins.enabled`

Entry-point plugins are gated by `plugins.enabled` in Hermes' `config.yaml`.
After installation, run:

```bash
hermes-vylen-gateway init
```

The command creates or updates the Hermes config and adds `vylen` to
`plugins.enabled` without changing platform settings.

## Platform Registration

`register(ctx)` registers a dynamic Hermes platform named `vylen`.

The adapter depends on Hermes' plugin platform registry accepting a registration
with:

- `name="vylen"`
- `adapter_factory`
- `check_fn`
- `required_env=["VYLEN_INSTANCE_TOKEN"]`
- `cron_deliver_env_var="VYLEN_HOME_CHAT_ID"`

`check_fn` returns true only when the gateway environment is valid. The gateway
config loader then auto-enables the platform for the current Hermes process.

## Required Adapter Methods

Hermes platform adapters implement the `BasePlatformAdapter` contract:

- `connect()`
- `disconnect()`
- `send(chat_id, content, reply_to=None, metadata=None)`
- `get_chat_info(chat_id)`

The Vylen adapter uses `connect()` to start the WebSocket supervisor. Inbound
client requests are not converted into Hermes `MessageEvent` objects; they are
handled through an in-process OpenAI-compatible request dispatcher in
`agent_runner.py`.

`send()` is the Hermes-to-Vylen push path. Hermes cron delivery and many media
helpers eventually call `send()` or a nearby adapter method. The plugin emits a
small `push` frame to Vylen Cloud and lets clients fetch media through
short-lived blob tokens when needed.

## In-Process Request Handling

Vylen Cloud still sends HTTP-shaped request frames over the gateway WebSocket.
The plugin handles those frames directly inside Hermes rather than forwarding
them to Hermes' loopback API server.

`agent_runner.py` mirrors the OpenAI-compatible Hermes API surface needed by
Vylen clients, including:

- health and capability endpoints;
- chat completions;
- responses;
- long-running runs and run events;
- run stop and approval flows.

This avoids local API-server configuration, avoids browser-origin issues, and
keeps the gateway as a single outbound connection from the Hermes process.

## Cron Delivery

Hermes resolves `--deliver vylen` only when the plugin registration declares a
cron delivery env var and that env var has a non-empty value. The plugin sets:

```bash
VYLEN_HOME_CHAT_ID=inbox
```

when the user has not provided a value. Vylen routes by the paired instance and
owning user, so the chat id is a compatibility bucket for Hermes' delivery
resolver.

Hermes may wrap cron output in a text envelope before calling `send()`. The
plugin strips the common envelope form and sends structured `cron_job_id` and
`cron_job_name` fields when they are available.

## Docker Notes

The common Hermes container layout is:

- Hermes home at `/opt/data`;
- Hermes virtualenv at `/opt/hermes/.venv`;
- Hermes binary at `/opt/hermes/.venv/bin/hermes`.

Container recreation can wipe packages installed into the virtualenv. If the
container is recreated, reinstall the package into that environment:

```bash
docker compose exec hermes bash -lc \
  'VIRTUAL_ENV=/opt/hermes/.venv uv pip install hermes-vylen-gateway'
```

For editable development against a checkout:

```bash
docker compose exec hermes bash -lc \
  'VIRTUAL_ENV=/opt/hermes/.venv uv pip install -e /path/to/hermes-vylen-gateway'
docker compose restart hermes
```

## Debugging

Run plugin discovery manually inside the Hermes environment:

```bash
python -c '
import logging
logging.basicConfig(level=logging.DEBUG)
from hermes_cli.plugins import discover_plugins
discover_plugins()
'
```

Common symptoms:

- `Skipping 'vylen' (not in plugins.enabled)` means run
  `hermes-vylen-gateway init`.
- `Plugin has no register() function` means the package entry point is wrong.
- `VYLEN_INSTANCE_TOKEN is not set` means the plugin is installed but not
  configured for a paired instance.

Check whether Hermes sees the platform:

```bash
python -c '
from gateway.config import load_gateway_config
cfg = load_gateway_config()
for p, pc in cfg.platforms.items():
    print(f"{p.value}: enabled={pc.enabled}")
'
```

The upstream Hermes implementation is available at
https://github.com/NousResearch/hermes-agent. Useful files to inspect there
include the plugin loader, platform registry, gateway config, gateway runner,
base platform adapter, and cron scheduler.
