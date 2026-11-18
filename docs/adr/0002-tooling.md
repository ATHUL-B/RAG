# ADR 0002: Python tooling

- Status: accepted
- Date: 2026-11-18

## Decision

- **Python 3.12**, managed by **uv** (`uv.lock` committed, reproducible installs).
- **pydantic v2** for domain models and config (validation + JSON serialization of artifacts).
- **ruff** for formatting and linting, **mypy --strict** for types, **pytest** + coverage.
- `make check` runs exactly what CI runs.
- Heavy dependencies (PDF parsers, embedding models, vector DB clients) are added only
  with the milestone that needs them, each justified in its PR or ADR.

## Consequences

- Strict typing catches interface mismatches between stages early, at some verbosity cost.
- Tests needing models or network are marked `slow` and excluded from the default run.
