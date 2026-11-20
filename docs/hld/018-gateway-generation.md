# HLD 018: Generation through an OpenAI-compatible LLM gateway

## Goal

The local 8B generator is the bottleneck (HLD 015-017). Someone with access to an LLM
gateway (Bifrost, LiteLLM, …) can run a stronger model on the same retrieval and send the
run folders back. This adds a **generic, tested** client for any OpenAI-compatible
`/chat/completions` endpoint, plus a runbook (`docs/runbooks/gateway-generation-run.md`).

## Lessons applied from the earlier gateway attempt (PR #14, closed)

| PR #14 | Now |
| --- | --- |
| Forced tool call for the answer; the gateway's translation lost 78% of answers | `response_format: json_schema` (strict) with our schema; **no tools**. A reply with only `tool_calls` is an error, never parsed as an answer |
| Company-internal URL hardcoded in code and configs | URL and key **only from env** (`LLM_GATEWAY_URL`, `LLM_GATEWAY_KEY`); configs record only the variable names |
| Gateway-specific header in code | `gateway_key_header` in config: `Authorization` (Bearer) or e.g. `x-bf-vk` |
| No tests; integrity issues found after a full run | Unit tests for payload, headers, refusals, tool calls and errors. Integrity checks (2.1a) catch mangling. The runbook makes a **3-question smoke test with 0 errors** a stop point |
| Budget guard only for `anthropic` | Guard applies to every non-Ollama provider |

## Design

- **`OpenAICompatibleLLMClient`** (`generation/llm_clients.py`):
  - messages are system + user, `temperature: 0`, `max_tokens`;
  - the answer is read from `message.content`;
  - `message.refusal` becomes a provider refusal;
  - usage comes from `prompt_tokens` / `completion_tokens`.
- **`urllib_post_json`** takes optional headers. HTTP errors include the status code and
  the start of the server's reply (e.g. "response_format not supported"), and **never a
  header value**. A test checks the key never appears.
- **Config:** `provider: openai_compatible` plus `gateway_*` fields.
  `configs/generation-gateway.yaml` has the **same retrieval as `generation-ollama.yaml`**
  (a config test enforces it), so only the generator changes. `model_id` is a
  `<placeholder>` the factory refuses to run with.

## Verification

End to end against **Ollama's own OpenAI-compatible endpoint** (`http://localhost:11434/v1`,
llama3.1:8b), 3 dev questions:

- 0 errors;
- the JSON-schema answer and `working` parsed correctly;
- token usage read;
- cost computed from the configured price.

A real gateway still has to pass the runbook's smoke test.
