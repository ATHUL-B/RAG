# HLD 008: Cross-encoder reranking

## Goal

Backlog 3.5. Rerank the first-stage candidate pool with a cross-encoder (study guide,
Part 10) and measure the precision gain against its latency cost. Models: ADR 0004.

Reference points (all with filters): dense recall@5 0.377; hybrid recall@5 0.360,
hit@50 0.887, pool_recall 0.856.

## Non-goals

LLM rerankers, late interaction (ColBERT), score cut-offs / refusal (needs calibration,
Phase 2), fine-tuning the reranker.

## Design

### `CrossEncoderReranker(Reranker)` (`retrieval/rerankers.py`)

- Lazy-loads `sentence_transformers.CrossEncoder(model, revision, device, max_length)`.
- `rerank(query, candidates, top_n)`: scores every `(query.text, chunk.text)` pair in
  batches, sorts by score (stable: ties keep first-stage order), returns the top `top_n`
  as `ScoredChunk(stage="rerank")`.
- Scorer injected behind a small `PairScorer` protocol so tests use a fake.

### `RerankingRetriever(Retriever)` (`retrieval/reranking.py`)

1. Ask the inner retriever (dense, or hybrid) for `depth` candidates (default 50).
2. Rerank them, return the top `k`.
3. `stage_candidates` keeps the inner stages and adds `"first_stage"` (the list given to
   the reranker), so traces show the rank of the evidence before and after reranking.

Composition: `FilteredRetriever(RerankingRetriever(HybridRetriever(...)))`: filters are
extracted once and reach every inner search.

### Per-stage latency

`RetrievalResult.timings_ms` (stage → ms) is filled by `RerankingRetriever`
(`first_stage`, `rerank`). The runner stores them in each trace and reports p50/p95/p99
per stage, because the reranker's cost is the whole question here (Part 14 latency budget).

### Depth

`depth` trades recall of the pool against latency (Part 10, "how deep to rerank"). The
baseline is 50; a depth sweep is a later ablation on a dev split.

## Config and experiments

```yaml
retrieval:
  rerank:
    enabled: true
    model_id: cross-encoder/ms-marco-MiniLM-L-6-v2
    revision: 233902d25c440f23af6f7d6e94d2946bac0bee0a
    depth: 50
    batch_size: 16
    max_length: 512
    device: auto
```

| Config | vs | Change |
| --- | --- | --- |
| `rerank-minilm.yaml` | `hybrid.yaml` | + MiniLM reranker |
| `rerank-bge.yaml` | `hybrid.yaml` | + bge-reranker-base |
| `dense-rerank-bge.yaml` | `filters.yaml` | + bge-reranker-base over the dense pool (does the hybrid pool matter?) |

## Test plan

Fake pair scorer: ordering by score, stable ties, top_n, empty input, batching; reranking
retriever: depth, k, stage candidates, timings, filters passed through. Runner: per-stage
latency summary. One `slow` test with the real MiniLM model.

## Outcome (2026-11-18)

| Run (all with filters) | recall@5 | nDCG@5 | MRR | recall@10 | latency p50 |
| --- | --- | --- | --- | --- | --- |
| dense (default) | **0.377** | **0.292** | 0.290 | 0.468 | 8 ms |
| hybrid | 0.360 | 0.295 | **0.300** | 0.440 | 12 ms |
| hybrid + MiniLM | 0.333 | 0.252 | 0.261 | **0.490** | 232 ms |
| hybrid + bge-reranker-base | 0.340 | 0.245 | 0.245 | 0.489 | 1,558 ms |
| dense + bge-reranker-base | 0.367 | 0.260 | 0.257 | 0.487 | 1,577 ms |

- No reranker improves top-5 (all paired-bootstrap ties); MiniLM's nDCG@5 drop is
  significant. Small, non-significant gains at @10.
- Diagnosis (3M FY2018 capex): the labelled cash-flow page is split into header-less table
  fragments ("Acquisitions, net of cash acquired 13 (2,023) …") that a cross-encoder scores
  near 0, while it ranks a segment table that *does* contain the answer ($1,577, p38) first.
  Two effects: incomplete page labels (a correct answer on an unlabelled page counts as a
  miss) and, mainly, chunks that lost their context (which statement, which years).
- Decision: keep the reranker implementation and configs, **disabled by default**; it costs
  220 ms-1.5 s per question for no measured gain. Re-evaluate after chunks carry context
  (3.1 table-aware parsing, 3.2 structural chunking, 3.7 contextual chunks).
