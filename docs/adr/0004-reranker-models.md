# ADR 0004: Cross-encoder rerankers to evaluate

- Status: accepted (2026-11-18)

## Context

Backlog 3.5. The hybrid candidate pool contains the evidence far more often than the top 5
does (hit@50 0.887 vs hit@5 0.407). A cross-encoder reads question and chunk *together*
(Part 10) and should move that evidence up. Constraints as in ADR 0003: local, free,
reproducible.

## Decision

Evaluate two local cross-encoders via sentence-transformers, revisions pinned:

| Model | Revision | Params | 50 pairs on MPS | Output |
| --- | --- | --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 23M | ~325 ms | logits |
| `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | 278M | ~1,685 ms | 0-1 |

Both truncate a (question, chunk) pair at 512 tokens. MiniLM is the cheap baseline,
trained on web search (MS MARCO); bge-reranker-base is larger and multilingual. The eval
decides which one, if any, earns its latency.

## Consequences

- Reranking becomes the slowest online stage by far (hundreds of ms to seconds per
  question vs ~10 ms for retrieval), so latency per stage is now reported.
- Scores are not comparable across models (logits vs probabilities): they are used for
  ordering only. A score cut-off (Part 10) needs per-model calibration later.
- Larger rerankers (bge-reranker-v2-m3, LLM rerankers) are possible later behind the same
  `Reranker` interface.
