# HLD 004: Embeddings, exact vector index and dense retrieval

## Goal

Backlog 1.5: embed every chunk, build a persistent exact vector index, and retrieve the
top-K chunks for a query (study guide, Parts 5-6). Decisions in ADR 0003.

## Non-goals

ANN indexes (4.2), BM25/hybrid (3.4), filters derived from the question (3.3), the eval
runner (1.6).

## Design

| Component | File | Role |
| --- | --- | --- |
| `Embedder` (revised) | `indexing/base.py` | Returns NumPy `float32` matrices (`embed_documents`) and vectors (`embed_query`); exposes `model_id`, `dimension`, `max_tokens`, `count_tokens`. |
| `SentenceTransformerEmbedder` | `indexing/embedders.py` | Lazy model load (construction is cheap), pinned revision, device `auto` (MPS → CUDA → CPU), batching, L2 normalization, query prefix. |
| `ExactVectorStore(VectorStore)` | `indexing/vector_store.py` | Matrix + parallel chunk list. `upsert` replaces chunks with the same ID. `search` = equality pre-filter mask → dot product → `argpartition` top-k. `save(dir)` / `load(dir)` with a manifest (model ID, dimension, count); loading with a different model is an error. |
| `IndexStage` | `indexing/index_stage.py` | Reads every `ChunkSet`, embeds in batches, counts chunks longer than the model limit (they get truncated), saves the index, reports throughput. |
| `DenseRetriever(Retriever)` | `retrieval/dense.py` | `embed_query` → `store.search(k, query.filters)`; candidates tagged `stage="dense"`. |
| `build_embedder` | `indexing/factory.py` | Config → embedder. |

Index location: `data/indexes/dense/<chunker_fingerprint>__<model_slug>/` containing
`vectors.npy`, `chunks.jsonl`, `manifest.json`. Keyed by both, so changing the chunker or
the model never silently reuses a stale index.

Filters: `{"company": "3M", "year": 2018}` = all keys must match exactly (pre-filter:
the top-k is computed only over allowed chunks, Part 8).

Ties in score are broken by insertion order (stable), so results are deterministic.

## Config

```yaml
embedding:
  model_id: BAAI/bge-small-en-v1.5
  revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
  query_prefix: "Represent this sentence for searching relevant passages: "
  batch_size: 64
  device: auto
```

## Test plan

Unit tests with a deterministic hashing bag-of-words fake embedder: store upsert/replace,
top-k order, filters, k larger than the pool, dimension/normalization checks, save/load
round trip and model mismatch; index stage over fake chunk sets; dense retriever ranks.
One `slow` test loads the real model and checks that a matching passage outranks a
non-matching one.
