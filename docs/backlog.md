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
| 2.1 | LLM clients (Claude, Ollama; ADR 0005), labelled-source prompt, cited structured answers, deterministic answer checks + cost guard (HLD 010). **First valid runs with a local model (HLD 015):** `working`-before-`answer` schema (1/8 → 5/8 numeric), better retrieval puts evidence in 79% of prompts but the 8B generator is now the bottleneck | 11 | done (local); Claude run pending a key |
| 2.1a | Output-integrity checks (reject empty/trivial answers and markup leaks), raw provider response in the trace, per-question ladder-step + integrity report in the generation runner. From the first full run, where 78% of outputs were lost ([analysis](../experiments/analysis/2026-11-19-first-generation-run.md)) | 11, 13 | done (HLD 010) |
| 2.2 | Correctness + faithfulness judges; calibrate on ~50 hand-labelled answers | 13 | todo |
| 2.3 | Refusal path (done in 2.1: `answerable: false` + provider refusals) + our own unanswerable questions in `evals/` | 11, 13 | todo |

## Phase 3: Improve one stage at a time (each = one ablation entry)

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 3.1 | Cleaning (headers/footers, hyphenation) and table-aware parsing | 3 | todo |
| 3.2 | Page-aware chunking + contextual headers (filing + page title block) (HLD 009): recall@5 0.377 → 0.427 (n.s. per step) | 4 | done |
| 3.3 | Metadata filters (company, year) from the question, with relaxation (HLD 006): recall@5 0.273 → 0.377 | 8 | done |
| 3.4 | BM25 index + Reciprocal Rank Fusion (HLD 007): top-5 tie with dense, candidate pool better (hit@50 0.820 → 0.887) | 9 | done |
| 3.5 | Cross-encoder rerankers (MiniLM, bge-base) over hybrid/dense pools (HLD 008, ADR 0004): no top-5 gain, +220 ms-1.5 s; kept, disabled. **3.5b:** local LLM pointwise reranker over the planner pool (HLD 014): dev 0.740, test 0.667 (significant); ~26 s/question | 10 | done |
| 3.6 | Query rewriting / decomposition. **Step 1 done:** rule-based planner (HLD 011), dev recall@5 0.427 → 0.484, test 0.426 → 0.481 (both significant). **Step 2 done:** local LLM planner (Ollama llama3.1:8b, HLD 012), dev 0.484 → 0.583 (significant), test 0.481 → 0.546 (n.s.; MRR significant), +2.3 s/question uncached; opt-in, not default. Step 3: weighting the original list in RRF (HLD 013) **rejected**: no weight helps; losses are a top-5 budget problem, pool_recall 0.976 | 7 | done (rules default; LLM planner opt-in) |
| 3.7 | Contextual chunks, parent-child retrieval | 4, 12 | todo |

## Phase 4: Production concerns

| # | Item | Guide | Status |
| --- | --- | --- | --- |
| 4.1 | Latency budget: p50/p95 per stage, parallel retrievers, caching | 14 | todo |
| 4.2 | Index tuning (HNSW params) + index-recall measurement at scale (MS MARCO / NQ) | 5, 14 | todo |
| 4.3 | Eval regression gate in CI on a small golden subset | 13 | todo |
| 4.4 | Cross-check retrieval code on BEIR (SciFact, FiQA) against published nDCG@10 | 13 | todo |

## Discovered / unscheduled

- **Generation (HLD 015):** citations land in `working`, almost never in `answer`; add a
  check (or prompt rule) for cited answers. Repetition loops inside `working` truncate the
  JSON (4-6 errors per run): add `maxLength` on the Ollama path. Error traces record 0
  tokens because usage is lost when parsing fails. Scale errors (thousands vs millions)
  are the commonest wrong-number cause: consider a units rule in the prompt.

- ~~**Priority after 3.6:** rerank the fused pool.~~ **Done (HLD 014):** local LLM
  pointwise reranker, dev 0.583 → 0.740, test 0.546 → 0.667 (both significant);
  cross-encoder still worse. **Next for it:** latency (~26 s/question): depth 10, parallel
  grading, smaller grader, or distil grades into a cross-encoder (Part 14).

- Parse audit: add a whitespace-insensitive coverage variant (annotator evidence has
  spaces stripped, e.g. `Cashandcashequivalents`, which deflates current coverage).
- 91 of 26,264 chunks exceed bge's 512-token limit (sized with tiktoken; bge splits numbers
  more finely) and are truncated. Options: size chunks with the embedder's tokenizer, or
  lower chunk_size. Measure on the eval.
- ~~**Before any tuning**: split the golden set into dev/test, tune on dev only.~~ **Done
  2026-11-20:** `evals/financebench-split.json` (96 dev / 54 test, grouped by company,
  `rag.evaluation.splits`); scripts default to `--split dev`.
- Hybrid: query-dependent weighting / routing (ID- and number-heavy questions → BM25), and
  stripping instruction boilerplate from questions before BM25 (Part 7).
- **Statistical power:** 150 questions cannot resolve ~0.03 differences (every single step
  in 3.2 is a tie), and dev now has only 96. Grow the evaluation set (own questions in
  `evals/` with hand-labelled evidence pages, more datasets) before fine-grained ablations.
- Rerankers hurt even with context (HLD 009): investigate why (512-token truncation of
  context + chunk, header dominating the pair, web-trained models on tables); try
  bge-reranker-v2-m3 or an LLM reranker (Phase 2 provider) before giving up on Part 10.
- Context headers: per-chunk context (nearest row/section label, not only the page title),
  LLM-written context (Anthropic contextual retrieval, 3.7), and why they hurt
  `metrics-generated`.
- **Priority after 3.5:** chunk context. Table chunks start mid-statement with no
  statement name or year columns, so neither embeddings nor cross-encoders can judge them.
  Next: 3.2 structural chunking with heading/statement path (+ 3.1 table-aware parsing),
  then 3.7 contextual chunks; then re-run the reranker ablations.
- Eval labels are page-level and incomplete (the same figure appears on several pages, e.g.
  3M capex in the segment table p38 and the cash-flow statement p59). Consider an
  answer-aware relevance check alongside page labels (Phase 2, with generation metrics).
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
