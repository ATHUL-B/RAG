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

### 2026-11-21: format/answer retry probe

The reviewer requested 2 tries for each of the 3 smoke questions in prose, JSON-in-prompt,
and tagged formats. The probe set `ok=yes` only for `finish=stop` and a non-empty prose
answer, a JSON object with the expected non-empty fields, or the required tags. It printed
output metadata only:

```text
question                format  try  finish           chars out_tok  ok  id
financebench_id_02987   prose   1    stop               646     344  yes ff957a0c-3551-4a7e-9c9d-2a458ae0d501
financebench_id_02987   prose   2    stop               651     360  yes 25a37b87-24ec-48fc-a95c-eb18a90e5e4b
financebench_id_02987   json    1    stop               617     313  yes a297d818-d650-4cf2-bd7e-7652e4737712
financebench_id_02987   json    2    stop               593     281  yes c36ead3d-13f7-497d-a52d-aebf25a31717
financebench_id_02987   tagged  1    stop               626     301  yes fa1eed9d-5c55-474c-bae0-d0f9263e276c
financebench_id_02987   tagged  2    stop               626     309  yes 528849e4-7490-4f33-9e6b-c99abaabb6e3
financebench_id_07966   prose   1    stop               978     420  yes c3f044f2-b454-4e8d-99ec-46731d352326
financebench_id_07966   prose   2    stop               925     417  yes 8099a97a-c76d-4f24-b8bc-bc7164ec895e
financebench_id_07966   json    1    stop               622     225  yes 4b410c45-c9a5-45c8-8b46-d5d7c387c201
financebench_id_07966   json    2    content_filter     591     218  no  d95a9eb6-ba71-4240-84d3-9a291802f5a6
financebench_id_07966   tagged  1    content_filter       0       -  no  37e37373-efae-4d8b-b4a6-cc77fcd99b38
financebench_id_07966   tagged  2    content_filter      84      39  no  e6558645-e326-4ef6-9ffc-160637ae3375
financebench_id_04735   prose   1    stop               797     289  yes b05242a0-73cf-4dd4-bd19-a05ebd743500
financebench_id_04735   prose   2    stop               810     291  yes 59389f4f-c489-437a-9f65-ad4a8339399b
financebench_id_04735   json    1    stop               713     313  yes 94f517c1-4927-403d-adc5-72acbfc00400
financebench_id_04735   json    2    stop               660     294  yes b8c6e4bb-f520-46e2-9d05-b50e42befbf4
financebench_id_04735   tagged  1    stop               580     199  yes eba41b14-5e90-4c53-970a-965cfed3641f
financebench_id_04735   tagged  2    stop               824     265  yes 6fbf4bcb-0ec3-4ade-881c-31df416d8ed6
```

Prose passed 6/6, JSON-in-prompt passed 5/6, and tagged passed 4/6. All three filtered
responses were for `financebench_id_07966`; its prose passed both times, JSON changed from
pass to filter, and tagged filtered both times. This points to answer-format/output
filtering rather than the input prompt alone, but the exact trigger remains unknown. No
structured format passed all 6 trials, so no tagged-answer mode was adopted and the full
evaluations remain stopped. The probe printed output tokens but not input tokens, so exact
total cost cannot be reconstructed; the reviewer estimated $0.15–$0.20 before the run.
