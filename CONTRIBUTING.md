# Contributing

Issues and focused pull requests are welcome.

## Development Setup

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

Keep changes small and include tests for behavior changes. The package should
remain importable without Hermes installed; Hermes-specific imports should stay
lazy or be isolated behind runtime checks.

## Pull Requests

- Describe the user-visible behavior change.
- Include test output.
- Avoid committing generated files, local virtualenvs, cache directories, or
  package build artifacts.
- Do not include real instance tokens, provider keys, local service account
  files, or private Hermes data.
