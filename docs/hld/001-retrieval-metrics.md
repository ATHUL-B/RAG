# HLD 001: Retrieval metrics

## Goal

Pure, trusted implementations of recall@K, precision@K, hit rate@K, reciprocal rank
(MRR when averaged) and nDCG@K (study guide, Part 13), plus averaging overall and
**per segment tag**. Every later experiment is judged by these numbers, so a bug here
silently invalidates all of them.

## Non-goals

- Mapping golden `SourceSpan`s to chunk IDs (backlog 1.2).
- Running the pipeline over a golden set, traces, failure ladder (backlog 1.6).
- Generation metrics (Phase 2).

## Design

`rag/evaluation/retrieval_metrics.py`, inputs are deliberately generic:

- `ranked: Sequence[str]`: IDs of retrieved items, best first. **May repeat.**
- `qrels: Mapping[str, int]`: relevance grade per ID (TREC "qrels"); grade > 0 is
  relevant. Binary labels use grade 1.

The IDs are "relevance units". For BEIR they are chunk/passage IDs. For FinanceBench
(backlog 1.2) each retrieved chunk will be translated to the evidence unit it covers
(e.g. `doc_id#page`), so two chunks from the same evidence page produce a repeated
ID. Duplicate handling therefore matters:

| Metric | Definition | Duplicates |
| --- | --- | --- |
| recall@K | distinct relevant IDs in top K / total relevant | counted once |
| precision@K | positions in top K holding a relevant ID / K | each position counts (each chunk *was* useful) |
| hit@K | 1 if any relevant ID in top K | n/a |
| RR | 1 / rank of first relevant (0 if none) | n/a |
| nDCG@K | DCG(gains of first occurrences) / IDCG(best K grades in qrels) | repeats gain 0, so nDCG ≤ 1 |

IDCG uses **all grades in qrels** (as trec_eval does), not only retrieved ones:
missing a relevant item must lower the score. DCG uses gain = grade and discount
log2(rank + 1), matching the guide's worked example.

Precision divides by K even when fewer than K items were returned (standard; a short
list is not rewarded).

Queries with no relevant items (unanswerable) have undefined recall/nDCG: the functions
raise `ValueError`, and the evaluator skips them for retrieval metrics (refusal is
measured in Phase 2).

`RetrievalEvaluator(ks)` computes all metrics for one query into a `QueryMetrics`
(item id, tags, `{"recall@5": ...}`); `aggregate(results)` returns a
`MetricsReport` with the overall mean and the mean per tag, with counts.

Config: `evaluation.ks` (default `[1, 3, 5, 10]`).

## Alternatives

- `ranx` / `pytrec_eval` libraries: correct and fast, but hide the arithmetic this
  project exists to learn. We can cross-check against one later (backlog 4.4).

## Test plan

Hand-computed cases from the guide: naive top-5 recall@5 = 1/3, precision@5 = 1/5,
RR = 1/4; final top-4 precision = 3/4; nDCG@4 = 2.893 / 4.262 ≈ 0.679. Plus edge cases:
K larger than list, empty list, duplicates, graded relevance, unanswerable, invalid K,
segment aggregation.
