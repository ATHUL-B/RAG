# HLD 014: Local LLM reranker (pointwise grading)

## Goal

With the LLM query planner, the evidence is in the fused candidate pool for 97.6% of dev
questions but in the top 5 for only 58% (HLD 013). Ranking within the pool is the
bottleneck. Cross-encoders hurt on the old pool (HLD 008/009). Here the question is
whether they still hurt on this richer pool, and whether a **local LLM** that reads
question and passage can do better (study guide Part 10).

## How deep to rerank: measured first

Fused recall of `planner-llm` on dev at depth k:

| k | 5 | 10 | 15 | **20** | 30 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| recall | 0.583 | 0.745 | 0.807 | **0.859** | 0.901 | 0.915 |

At depth 20 a perfect reranker could reach 0.859, and deeper adds little at linear cost.

## Design

- **`LLMPointwiseReranker`** (`retrieval/llm_reranker.py`): one short Ollama call per
  candidate, schema-constrained to `{"grade": 0|1|2|3}`:
  - 3 = contains a figure or fact the question needs (any one of several counts)
  - 2 = right company, period and statement but not the figure
  - 1 = same company, other topic or period
  - 0 = irrelevant

  Candidates are sorted by grade, and **ties keep the fused order**, so the reranker
  only promotes; it never scrambles.
- **Pointwise, not listwise:** small models are weak at reading 20 passages and ordering
  them (position bias, lost in the middle). One passage per call avoids that.
- **Prompt layout chosen on dev**, measured as P(evidence graded above non-evidence) on
  12 dev questions, top 10:

  | Ordering | P(evidence > non-evidence) |
  | --- | --- |
  | question first, then passage | 0.65 (*worse than doing nothing*) |
  | fused order, no reranker | 0.67 |
  | **passage first, question last** | **0.75** (chosen) |
  | passage, then a one-sentence reason, then grade | ~same as question-last, 4x slower |

  With the question first, the 8B model graded both evidence tables (balance sheet,
  statement of operations) 0 for a DPO question. With the question last, the model reads
  it right before answering.
- **Safety:**
  - Any failed grading call returns the fused order for that question, counted as a
    fallback ("reranker down → fused ranking, flagged", CLAUDE.md §9).
  - The prompt treats the passage as data, not instructions.
  - `num_ctx` is set explicitly (4096), because Ollama silently truncates longer
    prompts.
- **Cache:** grades per (question, chunk) in
  `data/cache/rerank/<model>-<hash(system + schema + layout)>.json`. Re-runs and other
  depths reuse them.
- **Config:** `rerank.kind: cross_encoder | llm` plus an `llm:` sub-section. The
  composition root builds the reranker (it knows the cache path), and the factory
  refuses `kind: llm` without one.

## Results (dev n=96, paired bootstrap vs `planner-llm` 0.583)

| Config | recall@5 | Δ | 95% CI | better/worse | nDCG@5 | MRR | Rerank p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `planner-llm` | 0.583 | | | | 0.481 | 0.495 | |
| `planner-llm-rerank-bge` (depth 20) | 0.510 | -0.073 | [-0.167, +0.021] | 10/20 | 0.364 * worse | 0.353 * worse | 0.64 s |
| **`planner-llm-rerank-llm`** (depth 20) | **0.740** | **+0.156** | **[+0.083, +0.234]** | **21/3** | **0.624 *** | **0.628 *** | **26 s** |

- **Cross-encoders still hurt** on the richer pool, so it is the model, not the pool.
  Web-trained MS MARCO-style rerankers do not read flattened financial tables.
- **The LLM reranker lifts every segment** on dev:

  | Segment | Before | After |
  | --- | --- | --- |
  | `evidence:multi` | 0.524 | 0.667 |
  | `evidence:single` | 0.600 | 0.760 |
  | `type:metrics-generated` | 0.767 | 0.867 |
  | `type:domain-relevant` | 0.515 | 0.603 |
  | `type:novel-generated` | 0.484 | 0.766 |
  | `reasoning:logical-reasoning` | 0.389 | 0.556 |

  It closes 57% of the gap to the depth-20 ceiling (0.859).
- **Losses (3):** the grader gave an evidence table 0 (CVS balance sheet p303, Best Buy
  10-Q p16). The Best Buy top page (p18, graded 3) may also hold the store count, a
  page-label gap.
- **Latency:** 20 grading calls × ~1.3 s gives **~26 s per question** on an M5 laptop,
  0 fallbacks. Cached re-runs cost ~0.

## Test confirmation (n=54, run once after choosing on dev)

| Compared with | test recall@5 | Δ | 95% CI | better/worse |
| --- | --- | --- | --- | --- |
| `planner-llm` (0.546) | **0.667** | **+0.120** | [+0.037, +0.204] **significant** | 8/1 |
| `planner-rules` (0.481) | 0.667 | +0.185 | [+0.065, +0.306] significant | 14/2 |
| `pages-context` (0.426) | 0.667 | +0.241 | [+0.111, +0.361] significant | 18/2 |

- nDCG@5 0.566 and MRR 0.569 are also significant against all three.
- `evidence:multi` 0.571, `type:metrics-generated` 0.800.
- 0 fallbacks; rerank p50 27 s.
- Unlike the planner alone (n.s. on test), **planner + LLM reranker generalises to
  held-out companies.**

## Decision

`planner-llm-rerank-llm` is the **best-quality retrieval** config (dev 0.740, test 0.667).
It is **not** made the project default in this change: it needs Ollama plus ~27 s per
question, and the generation configs and `make data` should keep working without a local
model. Whether generation evaluation should switch to it is a separate decision.

## Production notes (Part 14)

26 s is fine for offline evaluation and batch answering, not for an interactive product.
Levers, in order:

1. Grade fewer candidates: depth 10 still has a 0.745 ceiling.
2. Run grades in parallel (Ollama `OLLAMA_NUM_PARALLEL`).
3. Use a smaller or faster grader, or distil the grades into a fine-tuned cross-encoder.
4. Cache by chunk for repeated questions.
