# Hermes integration internals

What you need to know about how this plugin plugs into `hermes-agent`. Distilled from spelunking + a Docker dogfooding session; written down so the next person doesn't re-discover any of it.

The spec for the whole epic lives at [docs/specs/001-hermes-gateway.md](../../docs/specs/001-hermes-gateway.md). This file is the operational layer underneath that.

---

## How Hermes discovers plugins

Hermes scans plugins from three sources, in priority order (bundled < user < project):

1. **Bundled** — directories under `<hermes-repo>/plugins/<name>/` with a `plugin.yaml`. Auto-loaded.
2. **User** — directories under `~/.hermes/plugins/<name>/` with a `plugin.yaml`. Gated by `plugins.enabled` in `~/.hermes/config.yaml`.
3. **Entry-point** — pip-installed packages that declare `[project.entry-points."hermes_agent.plugins"]` in pyproject.toml. **Also gated by `plugins.enabled`.** This is how Vylen plugs in.

Discovery happens via `hermes_cli.plugins.discover_plugins()`, which is called by `gateway/run.py` at startup, by `hermes_cli/main.py` for the CLI path, and idempotently from `_apply_env_overrides` in `gateway/config.py`.

## The entry-point contract (the bit that bit us first)

Hermes's loader does:

```python
module = entry_point.load()
register_fn = getattr(module, "register", None)
register_fn(ctx)
```

`entry_point.load()` returns whatever the entry-point string points at:

- `"hermes_vylen_gateway"` → the module ✅
- `"hermes_vylen_gateway:register"` → the `register` *function* ❌ (then `getattr(fn, "register")` is None → "Plugin 'X' has no register() function")

**Rule:** entry-points in `[project.entry-points."hermes_agent.plugins"]` must reference the **module**, not the function inside it. The module must export a top-level `register` callable.

## `plugins.enabled` gates entry-point plugins

By default, even an entry-point plugin that's installed in the venv is **opt-in**. The user must list it in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - vylen
```

Caveat: `hermes plugins enable <name>` only recognises filesystem plugins (user/bundled directories with `plugin.yaml`). For entry-point plugins it errors "Plugin 'X' is not installed or bundled." You have to write `plugins.enabled` directly. We bootstrap this in the install recipe with a one-liner against `/opt/data/config.yaml`.

## The Platform enum

`gateway/config.py` defines `class Platform(Enum)` with the bundled platforms (TELEGRAM, DISCORD, …). Plugin platforms get dynamic members via `Platform._missing_`:

- `Platform("vylen")` works because our `register(ctx)` calls `ctx.register_platform(name="vylen", …)`, which registers in `platform_registry`. `Platform._missing_` then sees `platform_registry.is_registered("vylen")` and mints a pseudo-member.
- **There is no `Platform.GENERIC`** — don't try to default to it. Use `Platform("vylen")` in the adapter's `__init__`.

## What `ctx.register_platform` accepts

From `hermes_cli/plugins.py`:

```python
ctx.register_platform(
    name="vylen",                     # must match Platform("vylen") lookup
    label="Vylen",
    adapter_factory=factory_callable, # receives PlatformConfig → BasePlatformAdapter
    check_fn=callable_returning_bool, # gate: required env vars present?
    required_env=["VYLEN_INSTANCE_TOKEN"],
    install_hint="...",               # shown on missing-env error
    **entry_kwargs,                   # e.g. emoji, setup_fn, env_enablement_fn
)
```

`check_fn` is the auto-enable gate. The gateway config loader (`_apply_env_overrides`) walks `platform_registry.plugin_entries()`, calls each `check_fn()`, and for the ones that return True sets `config.platforms[Platform(name)].enabled = True`. So if `VYLEN_INSTANCE_TOKEN` is set, our platform auto-enables on `hermes gateway run`.

## `BasePlatformAdapter` abstract methods

Four abstractmethods on `gateway/platforms/base.py:BasePlatformAdapter` you must implement (the rest have defaults):

- `async connect() -> bool`
- `async disconnect() -> None`
- `async send(chat_id, content, reply_to=None, metadata=None) -> SendResult`
- `async get_chat_info(chat_id) -> Dict`

Useful concrete methods that may want overrides: `send_image`, `send_voice`, `send_video`, `send_document`, `edit_message`, `delete_message`, `create_handoff_thread`. (Checkpoint 6 here.)

## How inbound messages reach Hermes

Hermes platform adapters dispatch user input as a `MessageEvent` (defined in `gateway/platforms/base.py`):

```
event = MessageEvent(
    text="...",
    message_type=MessageType.TEXT,    # or VOICE / PHOTO / AUDIO / DOCUMENT
    source=SessionSource(platform="vylen", chat_id=..., user_id=..., chat_type="dm"),
    media_urls=["/local/cache/path.ogg"],  # for audio/image/file paths
    media_types=["audio/ogg"],
    reply_to_message_id=None,
    ...
)
await self.handle_message(event)
```

For audio specifically, Hermes's gateway runner auto-transcribes `event.media_urls` of `audio/*` via `tools/transcription_tools.py` (faster-whisper or OpenAI Whisper, gated by `stt.enabled` in config). The transcript is prepended to `event.text` as `[The user sent a voice message. Here's what they said: "..."]` before the LLM sees it.

## Cron-driven outbound push

`cron/scheduler.py` keeps an in-process weakref to each registered adapter and calls `adapter.send(chat_id, content)` when a scheduled job fires. The adapter's `send()` is the only path that needs to know how to deliver to its platform. There's no separate "push" API — same `send`.

For Vylen this means: when checkpoint 6 implements `send()` properly, cron output should "just work" by emitting a frame the cloud forwards to mobile/web. No extra plumbing.

---

## Docker / install operational notes

The Hermes container (image `nousresearch/hermes-agent`) is opinionated:

- **Hermes home** is `/opt/data` inside the container, bind-mounted from `~/.hermes/` on the host. Contains `config.yaml`, `state.db`, etc. Always mount this; deleting it loses sessions.
- **Hermes venv** is at `/opt/hermes/.venv/`. Lives in the container layer — **`docker compose up -d` recreates wipe it** when the compose spec changes.
- **No `pip` on PATH** — the image uses `uv`. To install into the venv: `VIRTUAL_ENV=/opt/hermes/.venv uv pip install -e /opt/vylen-gateway-plugin`.
- **`hermes` binary isn't on PATH globally** — it's at `/opt/hermes/.venv/bin/hermes`. From `docker compose exec` you usually need the absolute path: `docker compose exec hermes /opt/hermes/.venv/bin/vylen-gateway-pair ...`.
- **Process user is `hermes` (UID 10000)**, not the host user. The entrypoint drops privileges via `gosu`.
- **`extra_hosts: - host.docker.internal:host-gateway`** is required on Linux/WSL2 Docker for `host.docker.internal` to resolve to the host. Docker Desktop sets this automatically; rootless / Linux Docker doesn't.

## Editable install + iteration

The standard dev loop:

```bash
# One-time after any compose recreate:
docker compose exec hermes bash -c 'VIRTUAL_ENV=/opt/hermes/.venv uv pip install -e /opt/vylen-gateway-plugin'

# For pure .py source edits afterwards:
docker compose restart hermes      # picks up edits in ~3-5s
```

Two gotchas:

- **Stale `__pycache__`** in the bind-mount can shadow source edits when the host venv (used by pytest) writes `.pyc` files Python prefers over freshly-edited `.py`. Symptom: code change "doesn't apply" — the error stack references a line number that doesn't match the current source.
  Nuke: `find vylen/gateway-plugin/src -name __pycache__ -exec rm -rf {} +`
- **Container recreate wipes the venv install.** If `docker compose up -d hermes` runs after any compose change, you must re-run the `uv pip install -e ...` step. `docker compose restart hermes` preserves the install. Baking the install into the container `command:` is possible but adds complexity; we chose to keep it manual.

## Debugging recipes

Plugin not loading? Run discovery manually with debug logging:

```bash
docker compose exec hermes /opt/hermes/.venv/bin/python -c '
import logging; logging.basicConfig(level=logging.DEBUG)
from hermes_cli.plugins import discover_plugins
discover_plugins()
'
```

Watch for:

- `Skipping 'X' (not in plugins.enabled)` → add to config.yaml
- `Plugin 'X' has no register() function` → fix entry-point to point at module, not function
- `Failed to create adapter for platform 'X': ...` → adapter `__init__` error

Plugin loads but won't connect? Run the doctor:

```bash
docker compose exec hermes /opt/hermes/.venv/bin/vylen-gateway-doctor --keep-open 5
```

That bypasses the gateway runner entirely and dials cloud directly. If the doctor succeeds and `hermes gateway run` doesn't, the issue is in the gateway loader path; if the doctor fails, the issue is in the plugin / network / cloud.

Check what platforms Hermes thinks are enabled at runtime:

```bash
docker compose exec hermes /opt/hermes/.venv/bin/python -c '
from gateway.config import load_gateway_config
cfg = load_gateway_config()
for p, pc in cfg.platforms.items():
    print(f"{p.value}: enabled={pc.enabled}")
'
```

## Key file references (in `external/hermes-agent/`)

When you need to read upstream:

- [hermes_cli/plugins.py](../../../external/hermes-agent/hermes_cli/plugins.py) — `PluginManager`, `_load_entrypoint_module`, `discover_and_load`. The single source of truth for plugin loading.
- [hermes_cli/plugins_cmd.py](../../../external/hermes-agent/hermes_cli/plugins_cmd.py) — `hermes plugins enable / disable / list` commands.
- [gateway/platform_registry.py](../../../external/hermes-agent/gateway/platform_registry.py) — `PlatformRegistry`, `PlatformEntry`. How platforms get registered.
- [gateway/platforms/base.py](../../../external/hermes-agent/gateway/platforms/base.py) — `BasePlatformAdapter`, `MessageEvent`, `Platform` enum members.
- [gateway/config.py](../../../external/hermes-agent/gateway/config.py) — `load_gateway_config`, `_apply_env_overrides`. Where plugins auto-enable.
- [gateway/run.py](../../../external/hermes-agent/gateway/run.py) — `GatewayRunner`. Where `discover_plugins()` is called at startup (around line 3375).
- [cron/scheduler.py](../../../external/hermes-agent/cron/scheduler.py) — push path from cron jobs to `adapter.send`.

