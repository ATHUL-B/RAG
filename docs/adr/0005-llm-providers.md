# ADR 0005: LLM providers for generation

- Status: accepted (chosen by the project owner, 2026-11-18: "option 3, starting with Claude")

## Context

Phase 2 adds answer generation (study guide, Part 11). Options considered: Claude via the
Anthropic API, a local model via Ollama, or both behind the existing `LLMClient` interface.

## Decision

Both, behind `LLMClient`, selected by config; **Claude first**.

| Provider | Default model | Why | Cost |
| --- | --- | --- | --- |
| Anthropic API (`anthropic` Python SDK 2.x) | `claude-opus-5-5` | Strongest grounded answers and instruction following | $4 / 1M input, $20 / 1M output tokens (thinking is billed as output) |
| Ollama (local REST API) | `llama3.1:8b` (configurable) | Free, offline, data stays on the machine | $0, slower, weaker |

Claude specifics (from the current API reference, not memory):

- Thinking is always on for `claude-opus-5-5`; depth is controlled with
  `output_config.effort` (default `medium`, set explicitly in config).
- Structured output via `output_config.format` (JSON schema), so answers parse reliably.
- Server-side refusal fallbacks enabled by default (`fallbacks: "default"`, beta
  `server-side-fallback-2026-07-01`); `stop_reason == "refusal"` is still handled.
- Prompt caching is not used yet: the system prompt is below the minimum cacheable size.
- Credentials come from the environment (`ANTHROPIC_API_KEY`, or a local `.env` file that is
  gitignored); they are never written to configs, logs or traces.

The judge model for LLM-as-judge metrics is decided in the next step (backlog 2.2); the
guide recommends a different model family from the generator (Part 13).

## Consequences

- Every evaluation run that calls Claude costs money: the eval script estimates cost first
  and refuses to run above a budget unless raised explicitly.
- The retrieved document text is sent to Anthropic when the Claude provider is used.
- Provider differences (structured-output support, token accounting) are hidden behind
  `LLMClient`, so the generator and evaluation code are provider-neutral.
