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
| 1.2 | Evidence mapping: golden `SourceSpan` (doc + page) → chunk relevance units, + chunk audit (HLD 003) | 13 | done |
| 1.3 | Baseline PDF parser (page-preserving) + parsed-document artifacts + parser audit script (HLD 002) | 3 | done |
| 1.4 | Recursive chunker (400 tokens, 0 overlap) + chunk artifacts (HLD 003) | 4 | done |
| 1.5 | Embedder + vector store + dense retriever (ADR 0003, HLD 004) | 5, 6 | done |
| 1.6 | Eval runner + CLI; per-question traces; failure-ladder status (rungs 1-2) (HLD 005) | 13 | done |
| 1.7 | First `experiments/LOG.md` entry: baseline recall@5 = 0.273 per segment | 13 | done |

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
| 3.3 | Metadata filters (company, year) from the question, with relaxation (HLD 006): recall@5 0.273 → 0.377 | 8 | done |
| 3.4 | BM25 index + Reciprocal Rank Fusion (HLD 007): top-5 tie with dense, candidate pool better (hit@50 0.820 → 0.887) | 9 | done |
| 3.5 | Cross-encoder reranker over the hybrid pool (pool_recall 0.856); rerank depth vs. latency | 10 | todo |
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
- 91 of 26,264 chunks exceed bge's 512-token limit (sized with tiktoken; bge splits numbers
  more finely) and are truncated. Options: size chunks with the embedder's tokenizer, or
  lower chunk_size. Measure on the eval.
- **Before any tuning** (RRF k/weights, BM25 k1/b, thresholds): split the golden set into
  dev/test, tune on dev only. Tuning on all 150 questions overfits the evaluation.
- Hybrid: query-dependent weighting / routing (ID- and number-heavy questions → BM25), and
  stripping instruction boilerplate from questions before BM25 (Part 7).
- **Priority after 3.3:** with filters, 66.7% of top-5 slots are the right filing but the
  wrong page: within-document ranking is now the bottleneck → 3.4 (BM25 + RRF), 3.5
  (reranker), 3.2 (structural chunking).
- Filters: doc-type filter (10-K vs 10-Q vs earnings; 18.8% of slots are same-company
  other filings, often same year), and forward-looking questions ("expected in FY2023"
  lives in the FY2022 filing): an LLM query planner (3.6) could map intent to year.
- Cleaning (3.1): normalize `\xa0`, re-join mid-word line breaks; try `fontTools`
  (pypdf warns it is missing for some CFF fonts) and compare on the parse audit.

- Put `uv` on PATH for the user's shell (installed at `~/Library/Python/3.9/bin`).
- Consider pre-commit hooks running `make lint`.
