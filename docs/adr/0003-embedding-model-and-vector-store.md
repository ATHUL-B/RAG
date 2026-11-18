# ADR 0003: bge-small-en-v1.5 embeddings with an exact NumPy vector store

- Status: accepted (chosen by the project owner, 2026-11-18)

## Context

Backlog 1.5 needs a dense retriever. Constraints: learning project on a Mac, reproducible
results, no data sent to third parties by default, ~26k chunks of ≤ 401 cl100k tokens.

## Decision

**Embedder:** `BAAI/bge-small-en-v1.5` via sentence-transformers, run locally.
- Revision pinned: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- 384 dimensions, max sequence length 512, L2-normalized output (cosine = dot product).
- Queries get the model's retrieval instruction prefix
  `"Represent this sentence for searching relevant passages: "`; passages get none.
- Measured on an Apple-silicon Mac: ~103 chunks/s on MPS, ~42 on CPU. Longest chunk of
  `3M_2018_10K` = 414 bge tokens (no truncation).

**Vector store:** exact (brute-force) search in NumPy: one float32 matrix, a dot product
per query, metadata filters applied as a mask *before* ranking. ~26k × 384 × 4 B ≈ 40 MB.

## Consequences

- Free, offline after the first model download, deterministic given the revision.
- Exact search is the **ground truth** for measuring an approximate index later
  (index recall, backlog 4.2); it does not scale to millions of vectors, by design.
- A small English model: expect weaker results on financial jargon than larger models.
  Swapping it is a config change plus a re-index; the index records the model ID and
  revision and refuses to mix models.
- Adds PyTorch (large dependency) through sentence-transformers.
