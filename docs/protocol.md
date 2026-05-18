# Gateway Protocol

This document describes the public contract between `hermes-vylen-gateway` and
Vylen Cloud. It is intentionally a boundary contract, not a description of
Vylen Cloud internals.

## Transport

The plugin opens one outbound WebSocket to:

```text
{VYLEN_CLOUD_URL}/v1/gateway
```

`https://` cloud URLs become `wss://`; `http://` URLs become `ws://`.

The WebSocket request includes:

```text
Authorization: Bearer <VYLEN_INSTANCE_TOKEN>
```

The token identifies one paired Hermes instance. The token is configured in the
Hermes environment and is not sent to clients.

## Handshake

After the socket opens, the plugin sends:

```json
{
  "type": "hello",
  "instance_meta": {
    "plugin_version": "0.1.0",
    "hostname": "host",
    "python_version": "3.12.0",
    "hermes_version": "optional"
  }
}
```

Cloud replies with either:

```json
{
  "type": "ready",
  "instance_id": "inst_...",
  "user_id": "user_...",
  "server_time": "2026-05-18T00:00:00Z"
}
```

or:

```json
{
  "type": "error",
  "code": "TOKEN_INVALID",
  "message": "..."
}
```

## Request Frames

Cloud sends HTTP-shaped request frames:

```json
{
  "type": "request",
  "request_id": "req_...",
  "method": "POST",
  "path": "/v1/responses",
  "headers": {
    "content-type": "application/json"
  },
  "body": "base64-encoded bytes"
}
```

The plugin handles supported Hermes routes in-process and streams the result
back through response frames. Unsupported paths return a structured error.

## Response Frames

The plugin starts each response with headers:

```json
{
  "type": "response_headers",
  "request_id": "req_...",
  "status": 200,
  "headers": {
    "content-type": "text/event-stream"
  }
}
```

Body bytes are base64-encoded:

```json
{
  "type": "response_chunk",
  "request_id": "req_...",
  "data": "base64-encoded bytes"
}
```

Successful streams end with:

```json
{
  "type": "response_end",
  "request_id": "req_..."
}
```

Failures use:

```json
{
  "type": "response_error",
  "request_id": "req_...",
  "code": "HERMES_UNREACHABLE",
  "message": "..."
}
```

## Response Resume

For Hermes response streams that expose a response id, the plugin keeps a
short-lived local in-memory buffer. Cloud can ask the same plugin process to
resume by sending:

```json
{
  "type": "response_resume",
  "request_id": "req_...",
  "response_id": "resp_...",
  "after_cursor": 12
}
```

The plugin replays buffered chunks after the cursor and then tails live chunks
until the response completes. If the response id is unknown, expired, or the
plugin restarted, the plugin returns `RESUME_UNKNOWN`.

## Push Frames

Hermes-initiated messages, including cron output, are sent to cloud as:

```json
{
  "type": "push",
  "chat_id": "inbox",
  "text": "message text",
  "cron_job_id": "optional",
  "cron_job_name": "optional"
}
```

Generated image pushes may include:

```json
{
  "image_token": "short-lived-token",
  "image_mime": "image/png",
  "image_filename": "output.png"
}
```

Clients fetch image bytes through Vylen Cloud; cloud tunnels that request back
to the plugin, and the plugin serves only tokens it minted.

## Transcription And Memory Frames

The plugin also handles typed frames for audio transcription and memory control
operations. These frames are intentionally narrower than arbitrary local file
access. Memory operations are constrained to supported Hermes memory targets and
plugin-owned proposal/snapshot metadata.

## Compatibility

Frame type names are stable within a package release. New optional fields may be
added by Vylen Cloud or the plugin. Unknown frame types are ignored by older
plugin versions unless a specific route requires them.
