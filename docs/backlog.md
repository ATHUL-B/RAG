# Backlog

Ordered by the principle **measure first, then improve**: build a thin end-to-end
baseline with an eval harness, then improve one stage at a time (ablation ladder,
study guide Part 13). Status: `todo` / `in-progress` / `done`.

## Phase 0: Foundations

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 0.1 | Project skeleton: tooling, CI, domain models, stage interfaces, config, timing, logging | - | done |
| 0.2 | FinanceBench downloader + adapter to `GoldenItem` (ADR 0001) | 13 | done |

## Phase 1: Baseline (learning path step 1)

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 1.1 | Retrieval metrics (recall@K, precision@K, hit rate, MRR, nDCG@K) + per-segment aggregation, tested on the guide's worked examples (HLD 001) | 13 | done |
| 1.2 | Evidence mapping: golden `SourceSpan` (doc + page) → chunk IDs at eval time | 13 | todo |
| 1.3 | Baseline PDF parser (page-preserving) + parsed-document artifacts + parser audit script (HLD 002) | 3 | done |
| 1.4 | Recursive chunker (400 tokens, 0 overlap) + chunk artifacts | 4 | todo |
| 1.5 | Embedder + vector store + dense retriever (ADR: embedding model, vector DB) | 5, 6 | todo |
| 1.6 | `ingest` and `eval` CLIs; per-question traces; failure-ladder report (rungs 1-2) | 13 | todo |
| 1.7 | First `experiments/LOG.md` entry: baseline recall@5 per segment | 13 | todo |

## Phase 2: Generation and generation metrics

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 2.1 | LLM client (ADR: provider), context assembler with labelled sources, cited answers | 11 | todo |
| 2.2 | Correctness + faithfulness judges; calibrate on ~50 hand-labelled answers | 13 | todo |
| 2.3 | Refusal path + our own unanswerable questions in `evals/` | 11, 13 | todo |

## Phase 3: Improve one stage at a time (each = one ablation entry)

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 3.1 | Cleaning (headers/footers, hyphenation) and table-aware parsing | 3 | todo |
| 3.2 | Structural chunking with heading paths; size sweep 200/400/800 ± overlap | 4 | todo |
| 3.3 | Metadata filters (company, year, doc type); version-trap questions | 8 | todo |
| 3.4 | BM25 index + Reciprocal Rank Fusion | 9 | todo |
| 3.5 | Cross-encoder reranker; rerank depth vs. latency | 10 | todo |
| 3.6 | Query rewriting / decomposition for multi-part questions | 7 | todo |
| 3.7 | Contextual chunks, parent-child retrieval | 4, 12 | todo |

## Phase 4: Production concerns

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 4.1 | Latency budget: p50/p95 per stage, parallel retrievers, caching | 14 | todo |
| 4.2 | Index tuning (HNSW params) + index-recall measurement at scale (MS MARCO / NQ) | 5, 14 | todo |
| 4.3 | Eval regression gate in CI on a small golden subset | 13 | todo |
| 4.4 | Cross-check retrieval code on BEIR (SciFact, FiQA) against published nDCG@10 | 13 | todo |

## Discovered / unscheduled

- Parse audit: add a whitespace-insensitive coverage variant (annotator evidence has
  spaces stripped, e.g. `Cashandcashequivalents`, which deflates current coverage).
- Cleaning (3.1): normalize `\xa0`, re-join mid-word line breaks; try `fontTools`
  (pypdf warns it is missing for some CFF fonts) and compare on the parse audit.

- Put `uv` on PATH for the user's shell (installed at `~/Library/Python/3.9/bin`).
- Consider pre-commit hooks running `make lint`.
