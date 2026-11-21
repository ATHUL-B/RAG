# HLD 023: Parent-child retrieval (backlog 3.7)

## Goal

Study guide Parts 4 and 12: *index small chunks for precise matching, return the larger
parent for context.* HLD 022 measured both halves of the trade-off:

- **Small chunks find more pages:** with 200-token chunks, page-level recall at 2,000
  prompt tokens was 0.656, vs 0.484 for 400-token chunks.
- **Small chunks make worse answers:** llama3.1:8b got 13/30 numeric answers right
  from 400-token chunks and 5/31 from 200-token fragments, at the same prompt budget.

Parent-child tries to get both: **search** the 200-token children, **return** the 400-token
parent that contains the best child.

## Design

- **`map_children_to_parents(children, parents)`** maps each child to the parent in the
  same document with the largest character overlap. Both chunkers are page-bounded and
  read the same parsed text, so the offsets line up. It is a pure function, with tests.
- **`ParentChildRetriever(children, parent_of, fanout)`**:
  - fetches `k * fanout` children (several share a parent), maps them to parents and
    de-duplicates;
  - returns the top *k* parents in best-child order, scored by their best child;
  - keeps the children in `stage_candidates["children"]`, so a trace shows which small
    chunk found each parent;
  - lets filters pass through, because child chunks carry the same metadata.
- **The parents are exactly the default 400-token chunks (same IDs).** So recall@5 compares
  directly with `planner-rules`, the LLM reranker's grade cache is reused, and the prompt
  size is unchanged (top 5 ≈ 1,616 vs 1,644 tokens).
- **Config:** `retrieval.parent_child` with `enabled`, `child_chunk_size` (200),
  `child_chunk_overlap` (0) and `fanout` (4).
  - The child index is the HLD 022 `chunks-200-o0` index. `load_children()` refuses to run
    if it's missing and says how to build it.
  - Dense mode only, for now (a config error otherwise).
- **Configs:** `planner-rules-pc.yaml` and `planner-llm-rerank-llm-pc.yaml`, each changing
  only `parent_child` (a config test enforces it).

## Not built: LLM-written chunk context (contextual retrieval)

Anthropic-style contextual retrieval asks an LLM to write a short context for every chunk,
given the whole document. Estimated for this corpus:

| Variant | LLM calls | Input tokens | On this Mac (llama3.1:8b) |
| --- | --- | --- | --- |
| Whole document per chunk (the original method) | 31,259 | ~3.7B | impossible: documents are a median 107k tokens, beyond the model's window |
| Page window (~1,500 tokens) per chunk | 31,259 | ~59M | **~47 hours** |

The chunks already carry a deterministic context header (company, filing, year and the
page's title block; HLD 009). An LLM-written context needs a paid API run with an approved
budget, or a much smaller corpus. Recorded as an option, not done.

## Results

Dev, n=96, paired against the same config without parent-child.

| | `planner-rules` | `planner-rules-pc` | `planner-llm-rerank-llm` (best) | `…-pc` |
| --- | --- | --- | --- | --- |
| recall@5 | 0.484 | 0.479 (−0.005, 5/7) | 0.740 | **0.781** (+0.042, CI [−0.026, +0.109], 9/7) |
| hit@5 | 0.521 | 0.510 | 0.781 | **0.844** (+0.062, CI [−0.010, +0.135], 9/3) |
| MRR | 0.405 | 0.375 | 0.628 | 0.617 |
| recall@10 | 0.630 | 0.677 (+0.047, n.s.) | 0.833 | 0.849 |
| **pool recall** (evidence anywhere in the returned list) | 0.760 | **0.885** | 0.976 | 0.983 |
| recall @2,000 prompt tokens | 0.484 | 0.490 | | |
| retrieval p50 / stack load | 15.8 ms / 1.2 s | 23.2 ms / 7.1 s | | |
| grading calls (new parents in the top 20) | | | | 656 new, 1,264 cached |

- **Children find more of the right parents, but don't rank them better.** In the default
  config the pool of returned parents holds the evidence much more often (0.760 → 0.885),
  but the top 5 doesn't change. The best child's similarity is no better a ranking signal
  than the parent's own.
- **With the LLM reranker on top, the larger pool helps:** recall@5 +0.042, hit@5 +0.062
  (9 better / 3 worse). This is the direction HLD 022 predicted (search small, read big),
  but at n=96 the intervals include 0.
- **Cost:**
  - +7 ms per query, because the child index has 59k vectors instead of 31k;
  - +6 s at start-up, to load the second index and build the mapping;
  - a second index on disk;
  - nothing extra at generation, since the prompt size is the same.

## Decision

- **Opt-in, not default.** Neither config improves significantly. The best-config gain
  (+0.042 / +0.062) is the most promising result since the LLM reranker, but by the project
  rule a non-significant gain doesn't change the default.
- **Not run on test.** Running the held-out split *to decide* whether to adopt a
  non-significant dev gain would be choosing by test. The right next step is more dev
  questions (backlog: statistical power), or combining it with the next change and
  re-testing on dev.
- **Kept:** `ParentChildRetriever`, the mapping, and both `-pc` configs.

## Lessons

1. **Pool recall and top-k recall measure different things.** Parent-child improved the
   pool a lot (+0.125) and the top 5 not at all, until a reranker could use the pool.
   Measure both (CLAUDE.md: candidate-pool recall as its own metric).
2. **Search small, read big works in the direction expected**, and the measured cost is
   small. With 96 questions, "promising" is as far as the evidence goes.

