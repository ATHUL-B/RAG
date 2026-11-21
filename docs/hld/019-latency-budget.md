# HLD 019: Latency budget for the best-quality retrieval (backlog 4.1)

## Goal

The best retrieval (LLM planner + LLM reranker, `planner-llm-rerank-llm.yaml`, dev recall@5
0.740) costs about 27 s per question, uncached (HLD 014). Part 14 says: **measure per
stage first, then parallelise independent work, cap rerank depth and cache, and give
each stage a budget.** This HLD does that, one change at a time. Every speed-up is checked
against dev recall@5, so none of it costs quality silently.

## Method

- **Cold and warm.** Every run since HLD 014 reads cached plans and grades, which measures
  the cache, not the model. A *cold* run uses the same config with `cache: false` on both
  LLM stages (planner and reranker), so every call goes to Ollama. *Warm* is the normal
  cached run. Both matter in production: cold is a new question, warm is a repeated one.
- **Per stage, p50 and p95**, from the traces' existing `timings_ms`:
  - `rewrite`: rules plus the LLM planner;
  - `first_stage`: filtered dense search for every query in the plan, fused with RRF;
  - `rerank`: the LLM grades;
  - `retrieve`: end to end.
- **Quality gate:** dev recall@5 and MRR, with a paired comparison against the baseline
  (`compare_runs.py`). A step that changes no grades must reproduce the baseline's top 5
  exactly.
- **Same machine and model throughout:** Apple M5, 16 GB, Ollama 0.40.16, `llama3.1:8b`,
  100% GPU. Latency numbers are only comparable within this table.

## Profile of one grading call (before any change)

16 real grading prompts (2 dev questions × top 8 candidates), warm model:

| Part | Time | Tokens |
| --- | --- | --- |
| reading the prompt (`prompt_eval`) | 704 ms | 588 |
| writing the grade (`eval`) | 390 ms | ~10: `{"grade": 3}` under the JSON grammar |
| total | **1,106 ms** | |

| Requests in flight | Wall time for 16 | Speed-up | Grades identical |
| --- | --- | --- | --- |
| 1 | 17.7 s | 1.00× | — |
| 2 | 9.2 s | **1.93×** | 16/16 |
| 4 | 9.1 s | 1.94× | 16/16 |

Grading is 20 independent calls per question, so it is the textbook case for Part 14's
"run independent work in parallel". **This probe turned out to be misleading:** the
server has a single slot, and the full pipeline showed no gain (step 1 below).

## Steps (one change each)

| # | Change | Quality risk | Check |
| --- | --- | --- | --- |
| 0 | Baseline, cold and warm | — | — |
| 1 | **Concurrent grading** (`rerank.llm.concurrency: 2`) | none expected: same prompts, same grades | throughput vs the baseline |
| 2 | **One-token grade**: the output schema is a bare integer `3`, not `{"grade": 3}` | small: the output format changes | dev recall@5, paired vs step 1 |
| 3 | **Rerank depth** 20 → 15 / 10 | real: fewer candidates graded | dev recall@5 from cached grades (no new calls) |

## Design

- **`LLMPointwiseReranker(concurrency=n)`** collects cached grades first, then grades only
  the missing candidates, up to `n` at a time on a thread pool.
  - Counters and cache writes stay on the calling thread; workers only call the model.
  - Failure semantics are unchanged: any failed grade means the first-stage order is
    returned for that question.
  - Sequentially, the first failure stops further calls. Concurrently, the batch in flight
    finishes first.
- **Config:** `retrieval.rerank.llm.concurrency` (default 1, so existing configs are
  unchanged). Any value above 1 helps only if the server has parallel slots
  (`OLLAMA_NUM_PARALLEL`).

## Results

### Measurement conditions (a lesson in itself)

The first cold baseline (96 questions) was **discarded**: the Mac went into idle sleep on
battery, and every sleep was counted as question time (only ~32 questions finished in an
hour). A second attempt lost a run to a closed lid. Latency runs now:

- run under `caffeinate -dims`, with the lid open;
- record the power source;
- repeat the baseline at the end of a series, to detect drift.

A latency number is only as good as the conditions it was measured in.

### Step 0: cold baseline (24 dev questions, awake, AC power)

| Stage | p50 | p95 |
| --- | --- | --- |
| `rewrite` (rules + LLM planner, 1 call) | 2.5 s | 3.4 s |
| `first_stage` (includes the planner) | 2.7 s | 3.9 s |
| **`rerank`** (20 grading calls) | **26.2 s** | 29.1 s |
| **`retrieve`, end to end** | **28.9 s** | 32.4 s |

- **The reranker is 90% of the latency**, at ~1.3 s per grading call in the full pipeline.
- Search itself (embedding, filters, RRF) is about 0.2 s.
- recall@5 on these 24 questions: 0.833.

### Step 1: concurrent grading: no gain on a single-slot server

| | Requests/s to Ollama |
| --- | --- |
| baseline (1 in flight) | 0.75 |
| `concurrency: 2` | 0.76 |

- Ollama on the dev machine runs **one slot** (`OLLAMA_NUM_PARALLEL:1`, `n_slots = 1` in
  its log): concurrent requests queue, and the server does the same work one after
  another.
- The 16-call probe above had suggested 1.93×. **The full pipeline disproved it.** Same
  lesson as the gateway filter in Phase 2: a small probe can mislead, and the real run decides.
- **Decision:** keep the code, tested, off by default (`concurrency: 1`).
  - It pays off only where the server really runs requests in parallel: Ollama with
    `OLLAMA_NUM_PARALLEL > 1`, or a hosted reranker.
  - We chose not to change the local Ollama setup.
  - In production this is a serving decision (slots, batching, GPU), not application code.

### Step 2: one-token grade: faster, but rejected

Cold, all 96 dev questions (`grade_format: integer`, its own new grade cache):

| | `{"grade": 3}` (baseline) | bare `3` |
| --- | --- | --- |
| rerank p50, same first 24 questions | 26.2 s | **19.6 s (−27%)** |
| retrieve p50, all 96 | — | 20.9 s |
| **recall@5, dev (n=96)** | **0.740** | 0.615: **−0.125, CI [−0.193, −0.057]** |
| MRR | 0.628 | 0.523 (significant) |

The grades show why. On the same 1,920 (question, passage) pairs:

| Grade | `{"grade": n}` | bare `n` |
| --- | --- | --- |
| 3 (contains the needed fact) | 261 | **9** |
| 2 | 493 | **1,494** |
| 1 | 13 | 0 |
| 0 | 1,153 | 417 |

Agreement: 48%. Forced to write a digit as its *first* token, llama3.1:8b collapses to
"2" and almost never says "3". **The "fixed" tokens `{"grade": ` aren't free filler.**
They are context the model reads before committing, the same lesson as the `working` field
in Phase 2 (HLD 015). The option stays in the code so the result is reproducible; the
default remains `object`.

### Step 3: rerank depth: a trade-off curve, not a free win

From the cached `{"grade": n}` grades, so no new calls and identical grades for the
candidates both depths share:

| Depth | Grading calls | Rerank cold (≈ calls × 1.3 s) | recall@5 | vs depth 20 | MRR |
| --- | --- | --- | --- | --- | --- |
| 0 (planner only, `planner-llm`) | 0 | 0 | 0.583 | −0.156 | 0.495 |
| 10 | 10 | ~13 s | 0.682 | **−0.057, CI [−0.115, −0.005]** | 0.582 |
| 15 | 15 | ~19.5 s | 0.708 | −0.031, CI [−0.073, +0.000] | 0.602 (significant loss) |
| **20** | 20 | **26.2 s** (measured) | **0.740** | — | 0.628 |

Each 5 candidates cost ~6.5 s and bring recall. Even depth 10 is still significantly better
than no reranker (+0.099). The evidence often sits at fused ranks 11-20, which is exactly
the "top-5 budget" problem HLD 013 found (pool recall 0.976). A smaller depth is a
legitimate **"balanced" setting** (~16 s, 0.682), but not a free speed-up.

### Warm vs cold

With every plan and grade cached, the same pipeline answers in **42 ms p50 / 56 ms p95**
(the depth runs above), about 700× faster than cold. That is Part 14's caching lever, at
its limit: it only helps for a question asked before, word for word.

## Latency budget (best-quality config, cold, dev machine)

| Stage | p50 | Share | Lever tried | Outcome |
| --- | --- | --- | --- | --- |
| LLM planner | 2.5 s | 9% | — | next: skip for simple questions |
| search (embed, filter, dense, RRF) | ~0.2 s | 1% | — | already small |
| LLM rerank, 20 calls | 26.2 s | 90% | concurrency / one-token grade / depth | no gain / rejected / trade-off |
| **total** | **28.9 s** | | | |

## Decision

- **No change to the best-quality config:** depth 20, `{"grade": n}`, concurrency 1.
  None of the three levers made it faster without costing recall on this machine.
- **Kept in code, off by default:** `concurrency` (needs a server with parallel slots) and
  `grade_format: integer` (rejected, kept for reproducibility).
- **The real lesson is where the time is:** 90% of the latency is one model reading 20
  passages one after another. Our code can't make that cheaper without changing what the
  model reads or which model reads it.

## Next

1. **A per-stage time budget** (Part 14: "give each stage a timeout"): if grading isn't
   finished within N seconds, return the fused order for the rest. This caps p95 instead
   of the mean, and the trace flags it.
2. **Skip the LLM planner** for single-fact questions the rules already handle (Part 14,
   "skipped for simple queries"): up to −2.5 s, measured against recall.
3. **A smaller or faster grader** (e.g. a 1-3B model), or **distilling** the LLM grades into
   the cross-encoder. That attacks the 1.3 s per call directly, but it is an ablation with
   real quality risk.
4. **Parallel slots** (`OLLAMA_NUM_PARALLEL=2`) if the local setup may change. The code is
   ready.
