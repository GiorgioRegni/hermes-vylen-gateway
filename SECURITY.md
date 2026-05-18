# Security Policy

## Reporting Vulnerabilities

Please report security issues privately by emailing security@vylenagent.com.

Do not open a public issue for vulnerabilities involving authentication,
instance-token handling, local file access, gateway frame validation, or Hermes
runtime data exposure.

## Security Model

- `VYLEN_INSTANCE_TOKEN` authenticates the outbound gateway WebSocket.
- Provider keys and Hermes-local credentials stay in the user's Hermes
  environment.
- Gateway blob tokens are short-lived and minted by the local plugin process.
- Response resume buffers are in-memory and local to the plugin process.
- The plugin should not introduce arbitrary local file read/write access through
  gateway frames.
