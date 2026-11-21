# HLD 022: Chunk size and overlap sweep (guide Part 15: "Overlap and size")

## Goal

The guide's learning path asks: *compare 200, 400 and 800 tokens, with and without 10%
overlap, and plot recall@5 against chunk size.* Every run so far used 400 tokens with no
overlap (HLD 003, chosen as a default, never measured). This sweeps all six combinations on
the default retrieval (`planner-rules`, no LLM). Each config changes only `chunk_size`,
`chunk_overlap` and a deeper `evaluation.ks` (to 20); a config test enforces this.

## A fair comparison needs three views

1. **recall@5**, the guide's metric. It's unfair across sizes: five 800-token chunks
   give the generator about 3× the text of five 200-token chunks.
2. **Recall at an equal prompt-token budget**: `recall_at_token_budget()` in
   `evaluation/retrieval_metrics.py`, with hand-computed tests, and `scripts/budget_recall.py`.
   It walks the ranking and keeps chunks while their prompt tokens (the compacted header +
   text, as the generator sees them) fit 1,000 / 2,000 / 4,000 tokens.
3. **Evidence text in the budget**: whether ≥ 80% of the annotator's evidence words are
   inside the in-budget chunks from the evidence page. FinanceBench labels are
   page-level, so recall counts any chunk on the right page as a hit, even the wrong half
   of it.

And the decider: **answers** (llama3.1:8b) at the same token budget.

## Results (dev, n=96)

| Chunks | chunks in corpus | evidence in one chunk | truncated by bge (> 512) | top-5 tokens | **recall@5** | recall @2,000 tok (page-level) | **evidence text @2,000 tok** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 / 0 | 58,971 | 34.9% | 69 | 890 | 0.469 | **0.656** (+0.172 *) | 0.344 |
| 200 / 20 | 60,084 | 34.4% | 76 | 905 | 0.500 | 0.609 (+0.125 *) | 0.312 |
| **400 / 0** (current) | 31,259 | 72.0% | 93 | 1,644 | 0.484 | 0.484 | 0.312 |
| 400 / 40 | 32,177 | 73.0% | 93 | 1,658 | 0.505 (+0.021) | 0.516 (+0.031) | 0.365 |
| 800 / 0 | 18,564 | 94.7% | **8,313 (45%)** | 2,924 | 0.516 (+0.031) | 0.443 (−0.042) | 0.385 |
| 800 / 80 | 18,669 | 94.7% | **8,565 (46%)** | 2,913 | 0.516 (+0.031) | 0.458 (−0.026) | 0.401 |

\* = paired 95% CI excludes 0 vs 400 / 0. No recall@5 difference is significant.

**Answers, at the same budget:** llama3.1:8b, `planner-rules` retrieval.

| | 400 × 5 sources (current) | 200 × 9 sources |
| --- | --- | --- |
| prompt tokens (all 96) | 180k | 187k |
| evidence page in prompt | 0.52 | 0.65 |
| **numeric correct** | **13/30** | 5/31 |
| paired numeric: only 400 correct / only 200 correct | | **8 / 1** (sign test p ≈ 0.04) |
| over-refusal | 0.19 | 0.18 |

## What it shows

1. **At an equal number of chunks, chunk size doesn't matter** for recall@5 here: every
   difference is a tie. The guide's plot would be flat.
2. **At an equal token budget, small chunks *look* much better** (+0.172 page-level recall).
   The cause is the labels, not retrieval quality: many small chunks touch many pages.
   Measured on the evidence **text**, the advantage disappears (0.31-0.40 for every
   setting).
3. **Answers decide it, and small chunks lose:** 13/30 → 5/31 numeric, at the same token
   budget. More pages reach the prompt, but as fragments. A table split into 200-token
   pieces is a row here and a header there, and the 8B model can't reassemble it (the same
   failure as HLD 015 and 021).
4. **800-token chunks can't be tested fairly with this embedder.** bge-small reads 512
   tokens, so 45% of 800-token chunks are embedded only partly. Their slightly better
   recall@5 comes from bigger chunks, despite the truncation, not from better embeddings.
5. **Overlap 10% at 400 is a consistent but small, non-significant gain:**
   - recall@5 +0.021;
   - evidence text @2,000 +0.053;
   - recall@10 +0.026;
   - it costs 3% more chunks.

## Decision

- **Keep 400 tokens, no overlap.** No setting is significantly better on recall@5. The
  only significant answer-level result (200 tokens) is *worse*. 400 / 40 is a candidate for
  a later combined change, but not enough on its own to re-index every config and
  invalidate the grade caches.
- **Keep `recall_at_token_budget` and `scripts/budget_recall.py`:** chunk-size and top-k
  comparisons should be made at an equal prompt budget, not an equal chunk count.
- **800 tokens needs an embedder with a longer window** (e.g. bge-m3 at 8k tokens) to be
  tested properly. That's an ADR-level change, not part of this sweep.

## Lessons

1. **A metric can reward the wrong thing.** Page-level labels make "more, smaller pieces"
   look better. Always check one level closer to what the user needs: the text, then
   the answer.
2. **Compare at equal cost.** The same k means a different prompt size for different chunk
   sizes. Fix the budget, then compare.
3. **Know the model's limits:** an 800-token chunk in a 512-token embedder is half
   invisible.
