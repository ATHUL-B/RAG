# Bifrost gateway smoke: debugging notes (2026-11-20–21)

**Baseline:** `adfc7c4` (the clean `main` baseline before these changes)

**Model configured locally:** `bedrock/global.anthropic.claude-sonnet-5-5`
**Result:** smoke test failed; this is a transport/configuration investigation, not a usable
quality evaluation. No full evaluation was started.

## Setup and cost estimate

- `make install` completed.
- `make data` found the 84 PDFs, parsed documents, chunks, and dense index already present;
  all ingestion steps were skipped.
- `make check` after the code changes: formatting/lint/type checks passed and 534 tests
  passed (3 deselected).
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

A four-case diagnostic probe isolated the schema path. It printed metadata only and made
no reply text or credentials visible:

| Probe | Finish | Content chars | Input / output tokens | Request ID |
| --- | --- | ---: | ---: | --- |
| Hello, no schema | `stop` | 34 | 24 / 17 | `bb7b7118-7147-4e8a-8ca6-c836db56bf92` |
| Hello, JSON schema | `content_filter` | 0 | 563 / 0 | `494e8b52-c34b-46d2-a17a-d012f27322b0` |
| FinanceBench prompt, no schema | `stop` | 678 | 3440 / 326 | `304c6040-41a7-46ee-84f0-b57d861694ce` |
| Same FinanceBench prompt, JSON schema | `content_filter` | 0 | 3954 / 0 | `e45cf941-7df1-4b48-867e-1ea30132415a` |

Both no-schema controls completed, while both schema requests were filtered. This isolates
the failure to the JSON-schema `response_format` path (likely its Bifrost/Bedrock
translation), rather than the route alone or SEC prompt content. It does not prove which
component emitted the filter. The client now supports a config option to place the schema
in the system prompt and omit `response_format`; its existing parser still validates the
JSON.

At the configured $2/M input and $10/M output rates, the probe usage estimates to about
**$0.0194**, above the review comment's under-$0.01 estimate. The client does not log reply
text or credentials.

## Interpretation and limits

The observed failure is filtering on the schema requests. The gateway did not expose a
detailed filter reason, so this cannot be attributed to a specific term, prompt component,
or policy.

The runner recorded zero tokens and zero cost because the failed responses did not provide
usage counts. This is **not** evidence that the gateway incurred no charge. No accuracy,
refusal, or quality conclusions can be drawn from these traces.

Per the runbook's smoke-test stop point, neither the 96-question dev run nor the
26-question unanswerable run was started. The run folders are local and git-ignored:

- `runs/generation-gateway-dev-20261120T122407Z` — first attempt, temperature rejected
- `runs/generation-gateway-dev-20261120T123944Z` — three-question smoke after omitting temperature
- `runs/generation-gateway-dev-20261120T124326Z` — one-question diagnostic retry

## Follow-up

The two filtered request IDs are `494e8b52-c34b-46d2-a17a-d012f27322b0` and
`e45cf941-7df1-4b48-867e-1ea30132415a`. Ask the Bifrost/Bedrock owner to inspect these
events.

### 2026-11-21: rerun without `response_format`

Following review feedback, the gateway client gained `gateway_use_response_format: false`.
It places the JSON schema in the system message instead of sending the OpenAI-compatible
`response_format` field; the answer parser still validates JSON shape and usability.

The four-case probe's no-schema controls succeeded, while both schema requests filtered.
The runbook smoke was then repeated with `gateway_use_response_format: false` and still did
not pass: 1/3 answers was usable and numerically correct; the other 2/3 returned partial
JSON with `finish_reason=content_filter` and were rejected as schema mismatches. The
successful answer used 3,654 input and 250 output tokens, costing $0.009808 at the
configured rates. The failed traces contain no usage counts, so their cost is unknown.

This confirms the JSON-schema `response_format` path filters even benign and real prompts,
but it does not explain the two remaining filters without that field. Paired no-schema
probes for those two questions, without the schema instruction, would be needed to isolate
prompt-content filtering from model-output filtering. Per the runbook, no full evaluation
was started.
