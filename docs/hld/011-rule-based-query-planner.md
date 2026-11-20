# HLD 011: Rule-based query planner (backlog 3.6, step 1)

## Goal

Study guide Part 7: fix the question before searching. This step is the **free,
deterministic baseline** that a later LLM planner (sub-queries, better filters) has to beat.
Every rule was sized and tuned on the **dev** split only. Test was run once, to confirm the
chosen default.

## Why: what the dev misses showed

| Problem in the question | Dev questions | Not hit before |
| --- | --- | --- |
| Instruction boilerplate dominates the embedding ("Assume you are…", "Round your answer…", "If X is not a useful metric… explain why") | 25 | 18 |
| Abbreviations the embedder does not connect to filing text (PP&E, COGS, DPO, ROA, capex) | 14 | 12 |
| Two-digit fiscal years not read as years ("FY22", "between FY20 and FY21") | 8 | 6 |
| Forecasts: "expected in FY2023" is published in the **FY2022** filing, so the year filter hides it | 5 | 3 |

## Design

```
FilteredRetriever (filters from the ORIGINAL question; short FY years; forecast → prior year)
  └─ [RerankingRetriever]
       └─ MultiQueryRetriever: search(original) + search(rewrite) ─► RRF
            └─ Dense | BM25 | Hybrid
```

- **Filter rules** (`retrieval/filters.py`, `FilterConfig`):
  - `short_fiscal_years`: `FY22` / `FY 20` / `FY'21` → 2022 / 2020 / 2021. The FY prefix
    is required, because a bare "22" is usually a count or a percentage.
  - `forward_looking_prior_year`: if the question has a forecast cue (expect, forecast,
    guidance, outlook, anticipate, project), also allow the year before each mentioned
    year. Unlike the rejected `year_window: 1` (HLD 006), only forecast questions widen.
- **Rewriter** (`retrieval/rewriting.py`, `RuleBasedRewriter`, `RewriteConfig`). It is a
  pure function of the question text and never sees the corpus:
  - `strip_boilerplate`: drop instruction sentences and bracketed formatting asides, but keep
    the core question, definitions ("DPO is defined as: …", which name the line items) and
    the question part of a lead-in sentence ("Based on the balance sheet, **what is …?**").
    Statement mentions become the titles filings use ("P&L statement" → "consolidated
    statements of income operations"). If every sentence matched a rule, the original is
    returned, so the query is never emptied.
  - `expand_abbreviations`: append the spelled-out name of each abbreviation found
    (glossary of 16 terms). It adds names only, never facts (Part 7, rule 2).
- **`MultiQueryRetriever`** (`retrieval/multi_query.py`): searches the original question
  **and** the rewrite (`keep_original`, Part 7 rule 1), each `candidate_depth` deep, then
  fuses with RRF. Filters pass through unchanged. If the rewriter changes nothing, it is
  exactly the inner retriever (no extra search).
- **Traces:** `RetrievalResult.queries` and `QueryTrace.queries` record the search texts
  actually used. The reranker passes them through.

## Results (dev n=96 unless stated; paired bootstrap, 95% CI)

| Config | Change | recall@5 | Δ | CI | better/worse |
| --- | --- | --- | --- | --- | --- |
| `pages-context` | dev baseline | 0.427 | | | |
| `planner-fy` | + short fiscal years | 0.427 | +0.000 | [0, 0] | 0/0 (MRR +0.009, doc_hit 0.927 → 0.948) |
| `planner-forward` | + forecast → prior year | 0.438 | +0.010 | [0, +0.031] | 1/0 (fixes the only `filtered_out`) |
| `planner-clean` | + strip boilerplate (original + clean, RRF) | 0.469 | **+0.042** | [+0.010, +0.078] | **6/0, significant** |
| `planner-abbrev` | + expand abbreviations (original + expanded, RRF) | 0.432 | +0.005 | [0, +0.016] | 1/0 |
| `planner-rules` | all four (vs `planner-clean`) | 0.484 | +0.016 | [0, +0.042] | 2/0 |
| `planner-rules` | all four (vs dev baseline) | **0.484** | **+0.057** | [+0.021, +0.099] | **8/0, significant** |
| `planner-rules` | **test confirmation** (n=54) | **0.481** (from 0.426) | **+0.056** | [+0.009, +0.120] | **4/0, significant** |

- **Largest gains:** `type:metrics-generated` 0.550 → 0.700, `evidence:multi`
  0.143 → 0.262, `reasoning:numerical-reasoning` 0.365 → 0.459. These are the long,
  instruction-heavy calculation questions.
- **No change:** `type:domain-relevant` (0.309) and `reasoning:logical-reasoning`
  (0.000). Those are short "what drove X" questions with no boilerplate. They need
  better page ranking or decomposition, not cleaner text.
- **Short fiscal years:** the AMD "FY22" questions now filter to 2022 correctly. Recall is
  unchanged because AMD has only one filing in the corpus, so the company filter already
  selected it. The misses are page ranking within that filing.
- **Latency:** p50 8.5 → 15.8 ms, from two dense searches and one extra query embedding.
- **Decision:** `planner-rules` is the **new default**. It is significant on dev, confirmed
  on test, never made a question worse on recall, and is cheap. Generation configs and
  `make data` now use it (the chunks and index are unchanged).

## What is left for the LLM planner (3.6 step 2)

- **Decomposition:** multi-evidence questions are still only 0.262. One search per
  needed statement or line item ("Walmart FY2018 accounts payable", "Walmart FY2018 cost
  of sales") is the next lever.
- Forecast and "as of" phrasing beyond keyword cues, and companies missing from the alias
  catalog.
- It must beat `planner-rules` on dev, with cost and latency recorded.
