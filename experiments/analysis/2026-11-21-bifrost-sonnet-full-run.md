# Bifrost / Bedrock Claude Sonnet evaluation run (2026-11-21)

**Repository SHA:** `da25f518be245dcbfa6a176aaed20c5cfcbb0ca8` (current `main` at run time)
**Model:** `bedrock/global.anthropic.claude-sonnet-5-5`
**Gateway:** Bifrost OpenAI-compatible endpoint. The endpoint and virtual key are kept in
local environment variables and are intentionally omitted here.
**Retrieval:** unchanged FinanceBench configuration from `main`.
**Result:** the three-question smoke met the runbook's content-filter allowance, but the
96-question dev run exceeded the stop threshold by a wide margin. Treat full-run metrics as
diagnostic only; no quality comparison was recorded and the unanswerable run was not run.

## Setup

- `make install` completed.
- `make data` found the 84 FinanceBench PDFs and existing parsed documents, chunks, and
  index; downloads and ingestion were already complete.
- `make check` passed: 539 tests passed, 3 deselected; lint and type checks passed.
- The dev dry run assembled 96 prompts and estimated a maximum cost of **$1.23**. The
  26-question unanswerable estimate was **$0.33**. Dry-run mode sent no model requests.
- Gateway options used locally: `provider: openai_compatible`, model above,
  `gateway_key_header: x-bf-vk`, `gateway_temperature: null`,
  `gateway_use_response_format: false`, two content-filter retries (three attempts total),
  300-second timeout, and configured prices of $2/M input and $10/M output tokens.
  These settings are local and are not included as a config change in this report PR.

## Smoke run

Run: `generation-gateway-dev-20261121T065425Z` (3 questions)

- Errors: **1**, solely `content_filtered` after the configured retries.
- Two responses were usable: one correct and one refused. The remaining response was
  rejected as filtered.
- Usage reported: 10,088 input tokens and 668 output tokens; reported cost **$0.026856**.
- This met the runbook smoke allowance: only content-filter errors and at least two usable
  responses. It allowed proceeding to the full dev run.

## Full dev run

Run: `generation-gateway-dev-20261121T070405Z` (96 questions)

- Errors: **67**: 66 `content_filtered`, 1 `schema_mismatch`.
- Numeric match: **13/16 (81.25%)** among scored numeric cases. This is not a reliable
  quality result because the run had 67 errors.
- Reported tokens: 110,516 input and 8,787 output; reported cost **$0.308902**.
- Generation latency: p50 **4.29 s**, p95 **7.20 s**, p99 **13.33 s**, max **31.70 s**.
- Retrieval latency: p50 **70 ms**, p95 **199 ms**, p99 **333 ms**, max **378 ms**.

The runbook says to stop when a full run has more than three errors. Since this run had 67,
the 26-question unanswerable run was deliberately not started. The error rate also makes
the aggregate answer, citation, refusal, and numeric metrics unsuitable for comparing
models. In particular, no inference about refusal quality can be made without the
unanswerable split.

## Interpretation and next step

The endpoint, authentication header, model, and request format were sufficient to produce
usable answers. However, intermittent content filtering remained frequent even with
`response_format` omitted and retries enabled. The single `schema_mismatch` is separately
counted; this run does not establish its cause. The gateway did not provide a per-error
diagnosis in the summary, so the filter trigger cannot be attributed to a specific prompt,
answer format, or gateway component from these aggregate results alone.

These results extend the earlier transport and format probes documented in
[`2026-11-20-bifrost-gateway-smoke.md`](2026-11-20-bifrost-gateway-smoke.md). The next useful
step is to inspect the filtered and schema-mismatch entries in the archived traces with the
Bifrost/Bedrock owner, then repeat the full evaluation only after the filter behavior is
understood or materially reduced. No row was added to `experiments/LOG.md` because this was
not a valid quality evaluation.

## Artifacts

The run folders and archive are local, git-ignored artifacts and are not committed in this
PR. The archive at `/private/tmp/gateway-bifrost-runs-20261121.zip` contains only the valid
smoke and full dev run folders, each with its `summary.json` and `generations.jsonl`. It
contains no `.env` file or gateway credentials.
