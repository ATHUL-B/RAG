# HLD 020: A time budget for the LLM reranker (backlog 4.1, step 4)

## Goal

Part 14: *"track p50 and p95 per stage, and give each stage a timeout"*, and *"reranker
times out → use the fused first-stage order"*. HLD 019 found that the LLM reranker is 90%
of the best retrieval's latency (26.2 s p50 cold), and that no lever made it cheaper
without costing recall. A budget doesn't make the reranker faster. It **bounds** it: a
question never waits longer than the budget plus one call, and when the budget cuts
grading short, the system degrades gracefully and **says so in the trace**.

**What to expect.** On the dev machine the rerank time is uniform: p50 26.2 s, p95
29.1 s, because every question makes 20 calls. A budget below that cuts *every* question,
and behaves like a smaller rerank depth (HLD 019's trade-off curve). A budget above it is a
pure safety net for a slow or overloaded server. Both uses are measured below.

## Design

- **`LLMPointwiseReranker(time_budget_s=…, clock=…)`**:
  - **Cached grades are free** and always used.
  - Missing grades are requested **in first-stage order**, so the budget is spent on the
    most promising candidates first.
  - **Before each call** (or each batch, with `concurrency` > 1), the elapsed time is
    checked. Once the budget is spent, no new call starts. A call already in flight
    finishes, so the overrun is at most one call.
- **Ranking after a cut-off** (policy A, chosen by simulation below):
  - graded candidates come first, sorted by grade, with ties in first-stage order;
  - **ungraded candidates follow in first-stage order**, keeping their first-stage
    score and stage, so a trace shows they were never graded.

  A cut-off after k grades therefore behaves exactly like rerank depth k.
- **Degradation is visible**, which CLAUDE.md §9 requires ("reranker down → fused
  ranking, flagged in the trace"). This was only *counted* before:
  - `Reranker.last_degradation`: `None`, `"rerank_budget: graded k/n, rest in
    first-stage order"`, or `"rerank_failed: first-stage order"`;
  - `RetrievalResult.degraded` and `QueryTrace.degraded` carry it per question;
  - the run summary counts `budget_cutoffs` and `ungraded`.
- **Config:** `retrieval.rerank.llm.time_budget_s` (default `null`, no budget). The clock is
  injected, so unit tests are deterministic.
- **Per-call timeouts are separate:** the Ollama client's `timeout_s` (120 s) still bounds
  a single stuck call. For a hard ceiling, keep `timeout_s` well below the budget.

## Choosing the ranking policy: a simulation on cached grades (dev, n=96)

The reranker grades the 20 fused candidates in order, so a cut-off after k grades can be
simulated exactly from the cached grades. The simulator reproduces the real runs: k=20
gives 0.740, 15 gives 0.708, 10 gives 0.682, and 0 gives 0.583 (`planner-llm`). recall@5
when grading stops after k candidates:

| k graded | A: ungraded after all graded | B: ungraded before grades 0/1 | C: grades 2-3, then the rest in first-stage order |
| --- | --- | --- | --- |
| 0 | 0.583 | 0.583 | 0.583 |
| 3 | 0.583 | 0.604 | 0.583 |
| 5 | 0.583 | **0.635** | 0.583 |
| 8 | 0.646 | 0.651 | 0.656 |
| 10 | **0.682** | 0.651 | **0.682** |
| 12 | **0.688** | 0.646 | **0.688** |
| 15 | **0.708** | 0.651 | **0.708** |
| 20 | 0.740 | 0.740 | 0.740 |

- **B** helps only when very few candidates were graded (k ≤ 8). From k = 10 it is worse:
  first-stage ranks 11-20 hold evidence *less* often than candidates ranked 1-10 that the
  grader marked 0 or 1. The small model's low grades are a weak signal; first-stage
  position is a better one.
- **A and C tie from k = 8 up.** A is chosen because it's the simplest to explain: a
  cut-off after k grades is exactly rerank depth k.

## Results

### No-op check (warm, `time_budget_s: 60`, all grades cached)

recall@5 **0.740**, identical to `planner-llm-rerank-llm`; 0 cut-offs. Cached grades cost
no budget, and a budget that is never reached changes nothing.

### Cold, 13 s budget, all 96 dev questions (planner and grades uncached, awake, battery)

| | no budget (HLD 019, 24 q, AC) | **13 s budget** (96 q) |
| --- | --- | --- |
| rerank p50 / p95 / **max** | 26.2 s / 29.1 s / — | 13.7 s / 14.3 s / **14.6 s** |
| retrieve p50 / p95 / max | 28.9 s / 32.4 s / — | **16.3 s / 17.7 s / 19.6 s** |
| candidates graded per question | 20 | 9-16 (mean 11.5) |
| recall@5 (dev) | 0.740 | **0.688**: −0.052, CI [−0.104, −0.010] |
| MRR | 0.628 | 0.596 (significant loss) |
| vs fixed depth 10 (0.682) | | +0.005, CI [−0.026, +0.036]: tie |

- **The cap holds.** The slowest rerank was 14.6 s for a 13 s budget, so the overrun stayed
  under one call (≤ 1.6 s). End to end, p95 fell from 32.4 s to 17.7 s.
- **The quality cost is what the simulation predicted** (k ≈ 11-12 → 0.682-0.688).
  The budget doesn't beat the depth trade-off curve. It *is* that curve, expressed in
  seconds instead of candidates.
- **A budget adapts where a fixed depth can't.** Questions with short passages got up to
  16 grades in the same time; long ones got 9. On a slower machine or a loaded server, a
  fixed depth would get slower, while a budget would grade fewer and keep the promise.
- **All 96 questions were flagged in their traces** (`degraded: ["rerank_budget: graded
  11/20, …"]`). Nothing degraded silently.

## Decision

- **New config `planner-llm-rerank-llm-budget13.yaml`, the "balanced" profile:** about 16 s
  instead of about 29 s cold, recall@5 0.688 instead of 0.740. That is still +0.105 over
  the planner alone (0.583). The only change vs the best-quality config is `time_budget_s`.
- **The best-quality config is unchanged** (no budget). For production, a generous safety-net
  budget (e.g. 45 s, above the observed p95) costs nothing in normal operation (no-op
  check above) and bounds the bad days.
- **Not confirmed on test:** the knob only selects a point on the depth curve already
  measured on dev, so no new setting was tuned that test could contradict.

## Profiles after Phase 4.1 (dev, cold, dev machine)

| Config | recall@5 | retrieve p50 / p95 | Needs |
| --- | --- | --- | --- |
| `planner-rules.yaml` (default) | 0.484 | ~16 ms | nothing |
| `planner-llm.yaml` | 0.583 | ~2.7 s | Ollama |
| `planner-llm-rerank-llm-budget13.yaml` (balanced) | 0.688 | 16.3 s / 17.7 s | Ollama |
| `planner-llm-rerank-llm.yaml` (best quality) | 0.740 | 28.9 s / 32.4 s | Ollama |
| any of the above, repeated question (warm caches) | same | ~42 ms | caches |

## Next (rest of 4.1)

- Skip the LLM planner for simple single-fact questions (up to −2.5 s).
- A smaller or faster grader model, attacking the ~1.3 s per call directly (a quality ablation).

