# HLD 012: LLM query planner with a local model (backlog 3.6, step 2)

## Goal

Decompose questions that need several facts into **one focused search per fact** (study
guide Part 7, "decomposition"). The free rule-based planner (HLD 011) fixed noisy
questions but left `evidence:multi` at 0.262 and `type:domain-relevant` at 0.309 on dev.
One vector cannot sit near both the balance sheet and the income statement. A ratio
question needs both.

This step must **beat `planner-rules` on dev** (recall@5 0.484), with its latency recorded.
It runs on a local model through Ollama: no API key, $0, and data stays on the machine.

## Design

```
FilteredRetriever (rule-based filters from the ORIGINAL question: unchanged)
  └─ MultiQueryRetriever ─► RRF over:
       original question            (Part 7 rule 1: always kept)
       rule-cleaned question        (HLD 011)
       LLM sub-queries 1..4         (this HLD)
```

- **`LLMQueryPlanner`** (`retrieval/llm_planner.py`) is a `QueryRewriter`. It reuses the
  `OllamaLLMClient` from generation (`temperature: 0`, JSON schema via Ollama's `format`).
  The schema is `{"sub_queries": [string, 1..max]}`.
- **Prompt rules:** one query per distinct fact, line item or statement, each a short
  phrase with company, fiscal year and standard line-item wording. **Never include numbers
  or facts not in the question; do not answer.**
- **Validation:** whitespace normalised, deduplicated (case-insensitive), copies of the
  question dropped (compared *before* truncation), each capped at `max_chars`, at most
  `max_sub_queries`.
- **Failure means rules only.** A timeout, an API error or malformed JSON returns no
  sub-queries; it is counted in `planner.failures` and never cached.
- **Filters stay rule-based.** A wrong LLM-extracted filter would hide the evidence
  completely (Part 7 risk), so LLM filter extraction would be a separate, later ablation.
- **`CompositeRewriter`** runs the rules, then the planner, and drops duplicate texts.
- **Plan cache** (Part 7, rule 5):
  `data/cache/planner/<model>-<sha256(system prompt)[:12]>.json`. Re-runs are instant and
  identical, and editing the prompt automatically starts a fresh cache.
- **Observability:**
  - The `rewrite` stage time is in each trace's `timings_ms`, and `queries` lists every
    search text.
  - `metrics.json` records planner calls, cache hits and failures.
- **Preflight:** `evaluate_retrieval.py` checks that Ollama is reachable and the model is
  pulled before starting. Without it, a run whose every plan failed would look like a
  genuine "no improvement".

## Config

```yaml
retrieval:
  rewrite:
    strip_boilerplate: true
    expand_abbreviations: true
    keep_original: true
    llm_planner:
      enabled: true
      model_id: llama3.1:8b
      ollama_url: http://localhost:11434
      timeout_s: 120
      max_sub_queries: 4
      max_chars: 200            # also the schema's maxLength per sub-query
      max_output_tokens: 300    # Ollama num_predict: a runaway plan fails fast
      cache: true
```

`configs/planner-llm.yaml` differs from `planner-rules.yaml` only in the `llm_planner`
block (enforced by a config test).

## Evaluation plan

- Dev only: `planner-llm` vs `planner-rules`, paired bootstrap on recall@5, nDCG@5, MRR.
- **Expected:** `evidence:multi` and `reasoning:numerical-reasoning` up (sub-queries
  target each statement), `type:domain-relevant` possibly up ("what drove X" →
  "MD&A revenue drivers").
- **Risk:** RRF over up to 6 lists dilutes the original. If the planner hurts single-fact
  questions, the next knob is per-list fusion weights.
- **Latency:** the first run pays the LLM call per question (expected around 1-3 s on an
  M-series laptop). Cached re-runs pay about 0. Both are reported; the uncached number is
  what production would see without a cache.

## Results (llama3.1:8b on an Apple M5, paired bootstrap, 95% CI)

| Split | `planner-rules` | `planner-llm` | Δ recall@5 | CI | better/worse | MRR Δ |
| --- | --- | --- | --- | --- | --- | --- |
| dev (n=96) | 0.484 | **0.583** | **+0.099** | [+0.031, +0.172] **significant** | 16/4 | +0.090 * |
| test (n=54) | 0.481 | **0.546** | +0.065 | [-0.065, +0.194] n.s. | 11/5 | +0.105 * |

- **Biggest gain is decomposition, as designed:** `evidence:multi` 0.262 → 0.524 (dev) and
  0.357 → 0.536 (test). `type:domain-relevant` 0.309 → 0.515 (dev), where "what drove
  X" gets searches like "AMD FY2022 revenue drivers".
- **Latency:** planner p50 2.2 s, p95 3.2-3.6 s, max 4.5 s per question, uncached. With
  the plan cache the planner stage costs ~0 ms (a cached re-run reproduced dev recall
  exactly). Rules alone: ~16 ms end to end.
- **Audit of all sub-queries:**
  - No invented values. The only digits not from the question are "100" (a percentage
    formula) and "10" (from "10-K").
  - The model adds *prior years* for trend questions ("improving as of FY2022" → FY2020,
    FY2021 searches). This is reasonable, and it only changes search text: filters
    still come from the original question.
- **Losses** (4 on dev, 5 on test) share one mechanism. RRF over 5-6 lists lets 3-4
  agreeing sub-queries outvote the original question's correct page (e.g. JnJ "EPS
  expected to accelerate in FY2023": the earnings release p0 was rank 1 for the
  original, but sub-queries for FY2021-2023 EPS pulled in other pages).
- **A bug found by the first run:** one question looped inside a JSON string ("…if
  necessary, and to get correct revenue figure…") until the 120 s timeout, every time.
  The fix is `maxLength` in the schema, which Ollama enforces in its grammar, plus
  `num_predict` as a backstop. That question now takes 3.4 s and keeps its 3 good
  sub-queries. The fingerprint now covers system prompt + schema, so the fix could
  not reuse old plans. Runs before the fix (dev 0.589 with 1 timeout, test 0.537 with
  2) are superseded.

## Decision

**Not the default yet.** `planner-rules` stays the default for the generation configs and
`make data`. `planner-llm` is an opt-in config. Reasons:

1. The gain is significant on dev, but recall@5 is not significant on held-out
   companies (test, n=54).
2. It adds ~2.3 s per uncached question and makes Ollama a hard dependency.

Next ablations, one change each on dev:

- **Weight the original list higher in RRF**, which targets the losses above.
- LLM-extracted filters, kept separate (wrong-filter risk).
- Try a stronger or faster local model.

Re-run test only once a change is chosen.

## Test plan

A scripted fake LLM covers: sub-queries keep filters; cleaning, dedup, caps and
question-copy removal; every failure type falls back to no sub-queries and is counted;
cache hits, reload, and no caching of failures; cache file named by model and prompt
hash. Also tested: composite rewriter order and dedup; rewrite timing in traces; factory
composition and the error when a planner is missing; lazy `build_planner` (no network);
Ollama preflight (pulled, missing, unreachable).
