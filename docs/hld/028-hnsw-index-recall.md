# HLD 028: HNSW approximate search and index recall

## Goal

Backlog 4.2 (guide Parts 5 and 14). Every dense search so far has been **exact**: the
query is compared with all ~31k vectors (ADR 0003). That is correct by definition, and it
does not scale: cost grows linearly with the corpus. Production vector databases use an
**approximate** index, usually HNSW, which trades a little recall for far fewer
comparisons. Three questions:

1. **How does HNSW work?** Build one from scratch, readable, and check it against a
   reference library (faiss).
2. **What does approximation cost?** *Index recall@k*: the share of the exact top-k the
   approximate index returns. It is measured against exact search, not relevance labels,
   so it isolates the index from retrieval quality (CLAUDE.md §2: "measure index recall as
   its own metric").
3. **What happens with filters?** Our default pipeline hard-filters by company and year
   (Part 8). Filtered ANN is the classic failure: post-filtering an ANN result can return
   nothing.

## Design

- **`rag.indexing.hnsw.HNSWGraph`**: Malkov & Yashunin (2016). Each step of the
  paper is written out:
  - random levels `floor(-ln U / ln M)`;
  - greedy descent through the upper layers;
  - a beam search of width `ef` on layer 0 (Algorithm 2);
  - insertion with beam `ef_construction` (Algorithm 1);
  - the neighbour-selection heuristic (Algorithm 4), which keeps links pointing in
    different directions.

  `M` links per node per layer (`2M` on layer 0). The graph is NumPy vectors plus Python
  adjacency lists. It counts `distance_computations`: the cost HNSW saves.
- **Filtered search**: an `allowed` mask. The walk goes through every node, but only
  allowed nodes enter the results. The alternative it is compared with is
  **post-filtering** (search unfiltered, drop disallowed hits).
- **`HNSWVectorStore(VectorStore)`** wraps a loaded `ExactVectorStore`. If at most
  `exact_below` chunks pass the filters, it searches them **exactly**; otherwise it walks
  the graph with the mask. This is the fallback production databases use (for example
  Qdrant's "full scan threshold").
- **Config**: `retrieval.dense_index: {kind: exact | hnsw, m, ef_construction,
  ef_search, exact_below, seed}`. The default is `exact`, so all earlier runs are
  unchanged.
  - The graph is built in memory when the stack loads; it is not persisted. At 31k
    vectors the build takes seconds.
  - `ExactVectorStore` gains a read-only `vectors` view, and its `filter_mask` is now
    public.
- **Reference**: faiss-cpu `IndexHNSWFlat` (inner product), with the same `M` and
  `ef_construction`, in an optional `ann` dependency group (`uv sync --group ann`). The
  library code doesn't need it.
- **`scripts/index_recall.py`**: on FinanceBench (the default index, 31,259 chunks, 96
  dev questions) and FiQA (57,638 docs, 648 test queries, already embedded for HLD 025),
  for M ∈ {8, 16, 32} and ef ∈ {10 … 256}, it reports:
  - index recall@10, vectors compared per query, p50 single-query latency and build time,
    for exact, ours and faiss;
  - filtered recall with the planner-rules filters for each question: post-filter vs
    in-walk vs exact fallback.
- **End to end**: `baseline-hnsw` and `planner-rules-hnsw` change only `dense_index`.
  They show whether index recall < 1 moves retrieval recall@5 on dev.

## Results

All runs were local on 2026-11-22 and cost $0. Index recall@10 is measured against exact
search over all vectors. Latency is the p50 for a single query on one thread (`runs/index-recall-20261122T074139Z`).

### 1. Our HNSW matches faiss

At M=16 (selected rows; the full sweep, M ∈ {8, 16, 32} × ef ∈ {10 … 256}, is in the summary):

| Corpus | ef_search | Ours: recall@10 | Ours: vectors compared | faiss: recall@10 |
| --- | --- | --- | --- | --- |
| FinanceBench (31,259) | 10 | 0.793 | 260 | 0.739 |
| | 32 | 0.935 | 485 | 0.904 |
| | **64** | **0.984** | **761 (2.4%)** | 0.980 |
| | 256 | 0.995 | 2,130 | 0.992 |
| FiQA (57,638) | 10 | 0.750 | 281 | 0.744 |
| | 64 | 0.957 | 920 (1.6%) | 0.953 |
| | 256 | 0.992 | 2,685 | 0.991 |

- **The two implementations agree** within 0.03 at every point of both curves, so the
  from-scratch graph is correct. Faiss builds about 10x faster (FinanceBench: 1.7 s vs
  19 s; FiQA: 5.4 s vs 49 s) and queries 5-7x faster, because it is C++ and ours is
  Python. The recall curves are the same.
- **`ef_search` is the knob.** Recall rises steeply up to ef ≈ 64, then flattens. Each
  doubling of ef roughly doubles the vectors compared.
- **A bigger `M` helps at small ef** (FiQA ef=10: M=8 0.634, M=32 0.784). It costs more
  links per node, and at ef ≥ 128 all M converge.
- **FiQA needs a larger ef than FinanceBench** for the same recall (0.957 vs 0.984 at
  ef=64). The FinanceBench questions mostly land in a few company clusters.
- **Speed:** even our Python graph answers in 0.35 ms vs 1.9 ms for exact NumPy search
  over 31k vectors (FiQA: 0.44 vs 3.7 ms). Exact cost grows linearly with n; HNSW grows
  roughly with log n.

### 2. Filters: the exact fallback wins at our selectivity

All 96 dev questions get company/year filters. The **median filter leaves 442 of 31,259
chunks (1.4%)**; the maximum is 3,343. With `exact_below=2000`, 94/96 searches take the
exact fallback. At M=16:

| ef_search | Post-filter recall@10 | In-walk filter recall@10 | In-walk vectors compared |
| --- | --- | --- | --- |
| 10 | **0.551** | 0.905 | 731 |
| 32 | 0.889 | 0.994 | 1,223 |
| 64 | 0.980 | 1.000 | 2,075 |
| 256 | 0.999 | 1.000 | 14,056 (45% of the corpus) |

- **Post-filtering is the classic trap.** At small ef, half the true top-10 is lost,
  because most of the unfiltered neighbours belong to other companies.
- **Filtering inside the walk is accurate, but its cost explodes.** To fill a beam with
  allowed nodes it walks through disallowed ones: at ef=256 it compares 45% of the corpus.
- **The exact fallback is both exact and cheapest.** Comparing 442 vectors beats walking a
  graph to find them. Production databases do the same (a full-scan threshold, or
  per-tenant indexes).

### 3. End to end (dev, 96 questions, paired)

| Config | recall@5 exact → HNSW | 95% CI | recall@10 | retrieve p50 / p95 ms |
| --- | --- | --- | --- | --- |
| `baseline` (no filters) | 0.292 → 0.281 | [-0.031, +0.000] | 0.375 → 0.391 | 8.5 / 11.2 → **5.4 / 7.1** |
| `planner-rules` (filters) | 0.484 → 0.484 | identical | 0.630 → 0.630 | 16.0 / 19.1 → 16.7 / 20.4 |

- Without filters, an index recall of 0.98 changes one question in recall@5 (1 lost), and
  recall@10 moves within noise. This is not significant, and retrieval saves about 3 ms
  per question.
- With filters, the results are identical, because 94/96 questions use the exact fallback.
- **A bug caught by measuring.** The first HNSW run with filters was *slower* than exact
  (p50 22.8 vs 16.2 ms). The wrapper built the filter mask twice: once to decide on the
  fallback, and again inside the exact search. Building the mask costs 3.1 ms, a Python
  loop over 31k chunks' metadata, as much as the whole filtered search.
  `ExactVectorStore.search_within(vector, k, allowed)` now reuses the mask, and a test
  checks it is built once. The mask, not the dot products, is the next cost at scale; a
  payload index (company → row ids) would remove it.

## Decision

- **Keep exact search as the default.** At 31k vectors with 1.4%-selective filters,
  exact search is correct, fast enough (~3 ms), and HNSW would be bypassed for 98% of
  questions anyway.
- **HNSW is ready behind config** for a corpus where exact search becomes the bottleneck.
  Without filters that is around millions of vectors (exact cost ≈ 0.06 ms per 1k vectors
  here). Use `M=16, ef_search=64` (index recall 0.98 FinanceBench, 0.96 FiQA), and keep the exact
  fallback for selective filters. A real deployment would use faiss/hnswlib or a vector
  database rather than the Python graph, which exists to teach the algorithm.
- **Not done:** MS MARCO / NQ scale (the backlog originally asked for it). The user
  chose the corpora already embedded: an 8.8M-passage corpus is about a day of
  embedding on this Mac. The persisted graph and incremental inserts are also out of
  scope (the build takes seconds here).
