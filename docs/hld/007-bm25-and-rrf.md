# HLD 007: BM25 keyword retrieval and Reciprocal Rank Fusion

## Goal

Backlog 3.4. With metadata filters (HLD 006), **66.7% of top-5 slots are the right filing
but the wrong page**. Financial questions are full of exact terms and line-item names
("capital expenditure", "Purchases of property, plant and equipment", "dividends paid")
that dense embeddings blur. Add a BM25 keyword retriever and fuse it with dense retrieval
using RRF (study guide, Part 9), as one-change ablations on top of filters (recall@5 0.377).

## Non-goals

Learned sparse models (SPLADE), weighted score fusion, query routing, stemming,
persistent keyword index (it is rebuilt from the indexed chunks in a few seconds).

## Design

### Analyzer (`indexing/text_analysis.py`)

`analyze(text) -> list[str]`: lowercase; drop thousands separators inside numbers
(`1,577` → `1577`, so the number matches however it is written); tokens are alphanumeric
runs, keeping decimals (`7.9`, `3.25`); a small English stop-word list is removed. No
stemming (kept simple; candidate for a later ablation).

### `BM25Index(KeywordIndex)` (`indexing/keyword_index.py`)

Implemented directly (not a library) so the formula is visible and testable:

- Inverted index: term → (chunk positions, term frequencies) as NumPy arrays; chunk
  lengths; `avgdl`.
- `IDF(t) = ln((N − n_t + 0.5)/(n_t + 0.5) + 1)` (Part 9; never negative).
- `score(D, Q) = Σ_{t ∈ unique(Q)} IDF(t) · f(t,D)(k1+1) / (f(t,D) + k1(1 − b + b|D|/avgdl))`.
- Query terms are de-duplicated (repeating a word in the question does not double it).
- Filters: the same `domain.matches` pre-filter mask as the vector store (Part 8: "apply
  exactly the same filter to both").
- Only chunks with score > 0 are returned (sharing no term with the query = no match).
- `k1 = 1.2`, `b = 0.75` (guide defaults), from config.

### `KeywordRetriever(Retriever)` (`retrieval/keyword.py`)

Wraps the index; candidates tagged `stage="bm25"`.

### `ReciprocalRankFusion(Fuser)` (`retrieval/fusion.py`)

`RRF(d) = Σ_lists w_list / (k + rank_list(d))`, `k = 60` default; a chunk missing from a
list contributes 0. Weights are keyed by each list's `stage` name (default 1.0). Ties are
broken by the best individual rank, then by first appearance, so output is deterministic.
Output candidates are tagged `stage="rrf"` with ranks 1..k.

### `HybridRetriever(Retriever)` (`retrieval/hybrid.py`)

Runs each inner retriever with the **same query and filters**, `candidate_depth` deep
(default 50), fuses, returns the top k. `RetrievalResult` gains `stage_candidates`
(each inner retriever's ranked list) so the trace can say *which* retriever found the
evidence.

Composition: `FilteredRetriever(HybridRetriever([Dense, Keyword], RRF))`: filters are
extracted once and apply to both lists; relaxation applies to the fused result.

### Evaluation

Traces gain `stage_ranks`: for each stage, the rank of the first evidence chunk in that
stage's list (or null). Metric `pool_recall`: recall over the union of all stage lists
(the candidate pool a reranker would see, Part 13).

## Config and experiments

```yaml
retrieval:
  mode: dense            # dense | bm25 | hybrid
  candidate_depth: 50    # per retriever, before fusion
  bm25: {k1: 1.2, b: 0.75}
  fusion: {rrf_k: 60, weights: {dense: 1.0, bm25: 1.0}}
```

- `configs/filters.yaml`: current best (dense + filters), unchanged.
- `configs/bm25.yaml`: filters + `mode: bm25` (keyword only).
- `configs/hybrid.yaml`: filters + `mode: hybrid`.

## Test plan

Hand-computed BM25 (the guide's IDF and saturation numbers, length normalization),
analyzer cases (numbers, decimals, stop words), filters, empty/no-match queries. RRF on the
guide's worked example (order B, P, K, Q, A, R; k = 1 gives B = 0.750, P = 0.700), weights,
missing lists, ties. Hybrid: same filters to both, depth, stage candidates. Runner:
`stage_ranks`, `pool_recall`.

## Outcome (2026-11-18)

| Run (all with filters) | recall@5 | nDCG@5 | MRR | hit@50 (pool) |
| --- | --- | --- | --- | --- |
| dense | 0.377 | 0.292 | 0.290 | 0.820 |
| bm25 | 0.293 | 0.224 | 0.222 | 0.733 |
| hybrid (RRF) | 0.360 | 0.295 | 0.300 | 0.887 (union) |

- BM25 alone is significantly worse; hybrid vs dense at top 5 is a statistical tie
  (paired bootstrap, `evaluation/compare.py`).
- The lists are complementary: 10 questions are found only by BM25, and the union raises
  hit@50 from 0.820 to 0.887. Equal-weight RRF rewards agreement, so those single-list
  finds rarely reach the top 5.
- Decision: keep dense+filters as the top-5 default; use the hybrid pool as the input to the
  reranker (3.5), which is the guide's intended pipeline. No RRF/BM25 tuning on the test
  questions (needs a dev/test split first).
