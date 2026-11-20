# Bifrost gateway smoke: debugging notes (2026-11-20)

**Baseline:** `adfc7c4` (the clean `main` baseline before these changes)

**Model configured locally:** `bedrock/global.anthropic.claude-sonnet-5-5`
**Result:** smoke test failed; this is a transport/configuration investigation, not a usable
quality evaluation. No full evaluation was started.

## Setup and cost estimate

- `make install` completed.
- `make data` found the 84 PDFs, parsed documents, chunks, and dense index already present;
  all ingestion steps were skipped.
- `make check` passed before the client changes: formatting/lint/type checks passed and
  529 tests passed (3 deselected).
- The dev dry run assembled all 96 prompts and estimated cost at up to **$1.23** using the
  local Sonnet pricing configuration. It sent no model requests.
- The three-question smoke estimate was up to **$0.04** under the runbook's $1 cap.

## What failed

The first three-question attempt sent `temperature: 0`. Bifrost returned HTTP 400 from the
Bedrock route: `temperature` is deprecated for this model. The OpenAI-compatible client
now supports `gateway_temperature: null`, which omits the field while preserving the
deterministic default for other gateways.

After omitting `temperature`, the next three-question smoke still had **3 errors**:

| Question ID | Runner outcome | Observed response |
| --- | --- | --- |
| `financebench_id_02987` | `api_error` | Initially recorded as “gateway returned no content” |
| `financebench_id_07966` | `schema_mismatch` | `finish_reason=content_filter`; raw text was `{}` |
| `financebench_id_04735` | `api_error` | Initially recorded as “gateway returned no content” |

A one-question diagnostic retry for `financebench_id_02987` confirmed the response had
`finish_reason=content_filter`, message fields `content,role`, and no refusal field. The
client now preserves these safe metadata in its error message; it does not log response
text or credentials.

## Interpretation and limits

The observed failure is upstream content filtering: the response carries a
`content_filter` finish reason and no usable content. The response does not identify the
policy trigger. The diagnostic did not capture provider request IDs or a detailed filter
reason, so this cannot be attributed to a specific term, prompt component, or policy.
The third question's no-content response was not separately retried after adding the
diagnostic.

The runner recorded zero tokens and zero cost because the failed responses did not provide
usage counts. This is **not** evidence that the gateway incurred no charge. No accuracy,
refusal, or quality conclusions can be drawn from these traces.

Per the runbook's smoke-test stop point, neither the 96-question dev run nor the
26-question unanswerable run was started. The run folders are local and git-ignored:

- `runs/generation-gateway-dev-20261120T122407Z` — first attempt, temperature rejected
- `runs/generation-gateway-dev-20261120T123944Z` — three-question smoke after omitting temperature
- `runs/generation-gateway-dev-20261120T124326Z` — one-question diagnostic retry

## Follow-up

Ask the Bifrost/Bedrock owner to inspect the upstream filter event for the diagnostic
request and advise whether this model/route can answer FinanceBench prompts. Once the
filter behavior is understood, rerun the 3-question smoke and proceed to full runs only if
the runbook's pass criteria are met.
