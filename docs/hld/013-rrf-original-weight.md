# HLD 013: Weighting the original question in multi-query fusion (negative result)

## Goal

HLD 012's LLM planner lost 4 dev questions to `planner-rules`. The hypothesis was that RRF
over 5-6 lists lets agreeing sub-queries **outvote the original question's correct page**.
If so, giving the original list a higher RRF weight should recover those questions
without hurting others.

## Change

- `MultiQueryRetriever` labels each list with its search name (`original`, `rewrite_1`,
  …) before fusion. The factory builds RRF with `weights={"original": original_weight}`;
  every rewrite list has weight 1.
- `rewrite.original_weight` (default **1.0**: identical to before, verified on dev at
  0.583 to the digit, and by a unit test).
- Ablations change only that value vs `planner-llm.yaml`: 1.5, 2, 3.

## Results (dev n=96, cached plans, paired bootstrap vs `planner-llm` at 0.583)

| original_weight | recall@5 | Δ | 95% CI | better/worse | Losses recovered | Others made worse |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.583 | | | | 0/4 | |
| 1.5 | 0.594 | +0.010 | [0.000, +0.031] | 1/0 | 0/4 | 0 |
| 2.0 | 0.557 | -0.026 | [-0.073, +0.016] | 2/5 | 0/4 | 5 |
| 3.0 | 0.573 | -0.010 | [-0.062, +0.042] | 3/5 | 1/4 | 5 |

No value is significant, and the curve is not monotonic: noise. **Hypothesis rejected.**

## Why it failed: the trace says "budget", not "outvoting"

Per-list ranks of the evidence in the 4 lost questions:

| Question | Evidence rank in `original` | In the other lists |
| --- | --- | --- |
| Activision fixed asset turnover | 5 | 3, 2, 15, 15 |
| Activision 3-year capex % | 3 | 2, 2, 1, 3, 3 |
| JnJ "EPS expected to accelerate" | 3 | 7, 24, 33, 22 |
| Pfizer "spinning off segments" | 5 | 7, 20, 29, 9 |

The original never had the evidence at rank 1. It was at **rank 3-5, barely inside the
top 5 even under the rules**. More lists bring more strong pages that push marginal
pages out of the top 5. Weighting the original lifts its rank-1 and rank-2 pages, which
are mostly *not* the evidence, so it cannot help.

Meanwhile `pool_recall` (evidence anywhere in the fused candidates) is **0.976** with the
LLM planner, against recall@5 0.583. (The pool is larger now, up to 6 lists x 50 deep.)
Finding the evidence is mostly solved. **Ranking within the pool is the bottleneck.**

## Decision and next step

- Keep `original_weight: 1.0`. The configs stay as a record. Test was not run, because no
  change was chosen.
- Next lever: **rerank the fused pool**. The cross-encoders of HLD 008/009 hurt on the
  old, shallow pool. That should be re-measured on this much richer pool, and an LLM
  reranker (local model) compared against them. A larger context budget (top-8 or top-10
  into the prompt) is the generation-side alternative.
