# Phase 4.1 and the last Phase 3 items: latency, tables, chunk sizes, parent-child

Phase 3 part 2 ended with a best-quality retrieval (LLM planner + LLM reranker, dev recall@5
0.740, test 0.667) that took **~27 s per question**, and with open questions:

- Can it be made fast?
- Do flattened tables hurt?
- Was 400 tokens ever the right chunk size?
- Can small chunks help without fragmenting the answer?

This stretch answers them with **five measured changes** (HLD 019-023). Most of them
*didn't* change a default. That's the expected outcome of an honest ablation ladder, and
each "no" came with a reason.

Read the earlier walkthroughs first; this one uses their vocabulary (traces, the debugging
ladder, the dev/test split, paired bootstrap CIs, pool recall). Study-guide references look
like **(Part 14)**.

---

## 1. TL;DR

| # | Step | Backlog | Question it answered | Result | Kept? |
| --- | --- | --- | --- | --- | --- |
| 1 | **Latency budget**: cold vs warm, per stage (HLD 019) | 4.1 | Where do the 27 s go? | **28.9 s p50, 90% in the LLM reranker** (20 calls × ~1.3 s); warm 42 ms | method |
| 1a | Concurrent grading | 4.1 | Can independent calls run in parallel? | **no gain**: Ollama has 1 slot | code kept, off |
| 1b | One-token grade (`3` instead of `{"grade": 3}`) | 4.1 | Is the JSON wrapper wasted output? | rerank −27%, but **recall@5 −0.125** (significant) | **rejected** |
| 1c | Rerank depth 15 / 10 | 4.1 | Can we grade fewer? | −0.031 / −0.057: a trade-off curve | depth 20 kept |
| 2 | **Rerank time budget**, degradation in traces (HLD 020) | 4.1 | Can we bound latency? | 13 s budget: **p95 32.4 → 17.7 s**, recall 0.688 | **yes: "balanced" profile** |
| 3 | **Layout-mode parser** (HLD 021) | 3.1 | Do table rows on one line help? | retrieval ties; generation 10/32 → 8/31 (n.s.) | opt-in |
| 4 | **Chunk size × overlap sweep** (HLD 022) | 3.8 | Was 400 / 0 right? | recall@5 ties; 200 looks +0.172 at equal tokens, but **answers 13/30 → 5/31** | **400 / 0 kept** |
| 5 | **Parent-child retrieval** (HLD 023) | 3.7 | Search small, read big? | default tie; best config **0.740 → 0.781** (n.s.) | opt-in |

**In one sentence:** the latency is one 8B model reading 20 passages in a row, and it can be
bounded but not cheapened without losing recall. Ingestion changes that look obviously better
mostly tie, because llama3.1:8b's reasoning, not retrieval or parsing, now limits the answers.

---

## 2. What changed in the method

### 2.1 Cold vs warm, and measuring conditions

Every run since HLD 014 read cached plans and grades, so its latency measured the cache. A
**cold** run sets `cache: false` on both LLM stages. Two of the first three cold attempts
were **thrown away**: the Mac idle-slept on battery, then slept when the lid was closed,
and every minute of sleep was counted as question time. Latency runs now:

- run under `caffeinate -dims`, with the lid open;
- record the power source;
- **repeat the baseline at the end of a series**, to detect drift (it was 28.9 s on AC and
  30.3 s on battery).

`evaluate_retrieval.py --limit N` makes 24-question latency runs possible.

### 2.2 Compare at equal cost

recall@5 compares runs at the same *number* of chunks. Five 800-token chunks give the
generator about 3× the text of five 200-token chunks. **`recall_at_token_budget`** fixes the
prompt budget instead (step 4).

### 2.3 Read one level closer to the user

Page-level labels reward breadth: any chunk on the right page counts. Before any ingestion
decision, results were checked at two more levels:

- **evidence text:** is the annotator's text actually in the prompt?
- **answers:** does llama get the number right?

Twice (steps 3 and 4), the level closer to the user overturned the retrieval number.

### 2.4 Simulate before you spend

Grades are pointwise and cached, so a cut-off after *k* grades can be **simulated exactly**
from the cache. The simulator reproduced the real depth runs to three decimals, and it
chose step 2's ranking policy for free.

---

## 3. What changed in the code, at a glance

| Module | What it does | Step |
| --- | --- | --- |
| `retrieval/llm_reranker.py` | `concurrency`, `grade_format`, **`time_budget_s`** + injected `clock`, `last_degradation`, `budget_cutoffs` / `ungraded` counters | 1, 2 |
| `retrieval/base.py` | `RetrievalResult.degraded`; `Reranker.last_degradation` | 2 |
| `retrieval/reranking.py` | carries the reranker's degradation note into the result | 2 |
| `evaluation/retrieval_runner.py` | `QueryTrace.degraded` | 2 |
| `ingestion/pdf_parser.py` | **`PypdfLayoutParser`**, `normalize_layout_text`, plain-text fallback | 3 |
| `config/settings.py` | `ParserName.PYPDF_LAYOUT`, **`parsed_dir_for(parser)`**, `ParentChildConfig` | 3, 5 |
| `ingestion/chunkers.py`, `factory.py` | the parser joins the chunk fingerprint (non-baseline only) | 3 |
| `evaluation/retrieval_metrics.py` | **`recall_at_token_budget`** | 4 |
| `scripts/budget_recall.py` | budget recall over finished runs, paired CI | 4 |
| `retrieval/parent_child.py` | **`map_children_to_parents`**, **`ParentChildRetriever`** | 5 |
| `app.py` | `child_chunking`, `load_children` | 5 |

How the stack composes now:

```text
FilteredRetriever (company, year; relaxation)
 └─ RerankingRetriever (LLM grades top 20; optional time budget; degradation → trace)
     └─ MultiQueryRetriever (original + rule rewrite + LLM sub-queries; RRF)
         └─ Dense                         (default)
            | ParentChild(Dense over 200-token children) → 400-token parents   (opt-in)
```

---

## 4. Step 1: the latency budget (backlog 4.1, HLD 019)

### Measured before building

A profile of one grading call, on 16 real prompts with the model warm:

| Part | Time | Tokens |
| --- | --- | --- |
| reading the prompt | 704 ms | 588 |
| writing `{"grade": 3}` under the JSON grammar | 390 ms | ~10 |
| total | 1,106 ms | |

The cold baseline on 24 dev questions (awake, AC):

| Stage | p50 | p95 | Share |
| --- | --- | --- | --- |
| LLM planner | 2.5 s | 3.4 s | 9% |
| search (embed, filter, dense, RRF) | ~0.2 s | | 1% |
| **LLM rerank, 20 calls** | **26.2 s** | 29.1 s | **90%** |
| total | 28.9 s | 32.4 s | |

Fully cached, the same pipeline answers in **42 ms**: Part 14's caching lever, at its
limit, because it only helps for a question asked before, word for word.

### 1a. Concurrent grading: a probe that misled

The 20 grading calls are independent, the textbook case for Part 14's "run independent
work in parallel".

- **`LLMPointwiseReranker(concurrency=n)`** grades the missing candidates on a thread pool.
  Counters and cache writes stay on the calling thread; workers only call the model.
- **A 16-call probe showed 1.93× speed-up** with identical grades.
- **The full pipeline showed none:** 0.76 vs 0.75 requests per second. Ollama's log said
  why: `OLLAMA_NUM_PARALLEL:1`, `n_slots = 1`, so concurrent requests just queue. We chose
  not to change the local Ollama setup.
- **Kept, off by default:** the code helps only on a server with real parallel slots.
- **Lesson:** a small probe can mislead, and the full run decides (the same lesson as the
  gateway filter in Phase 2).

### 1b. One-token grade: the JSON wrapper isn't wasted

Ollama accepts a bare integer schema, so `3` costs 2 output tokens (~40 ms) instead of ~10
(~390 ms).

```text
INTEGER_GRADE_SCHEMA = {"type": "integer", "enum": [0, 1, 2, 3]}
```

| | `{"grade": n}` | bare `n` |
| --- | --- | --- |
| rerank p50, same 24 questions | 26.2 s | **19.6 s (−27%)** |
| recall@5, dev (n=96) | **0.740** | 0.615: **−0.125, CI [−0.193, −0.057]** |
| grade 3 / grade 2 (of 1,920) | 261 / 493 | **9 / 1,494** |

Forced to write a digit as its *first* token, the model says "2" for nearly everything.
**The fixed tokens `{"grade": ` are context the model reads before it commits**, the same
lesson as the `working` field in Phase 2. Rejected; the option stays in the code so the
result is reproducible.

### 1c. Depth: a trade-off curve

From cached grades, with no new calls:

| Depth | rerank cold (≈ calls × 1.3 s) | recall@5 | vs 20 |
| --- | --- | --- | --- |
| 0 (planner only) | 0 | 0.583 | −0.156 |
| 10 | ~13 s | 0.682 | **−0.057** (significant) |
| 15 | ~19.5 s | 0.708 | −0.031 (borderline) |
| **20** | 26.2 s | **0.740** | — |

The evidence often sits at fused ranks 11-20, so every 5 candidates buys recall.

---

## 5. Step 2: a time budget for the reranker (HLD 020)

### Problem

Part 14: *"give each stage a timeout"*, and *"reranker times out → use the fused first-stage
order"*. Step 1 showed the reranker can't be made cheaper, but it can be **bounded**.

### Code and intention

```text
for batch in batches(missing grades, size=concurrency):     # in first-stage order
    if time_budget_s and clock() - start >= time_budget_s:
        break                                               # no new call starts
    grade the batch (cached grades are free and used first)
ranked = graded (by grade, ties in first-stage order) + ungraded (first-stage order)
last_degradation = "rerank_budget: graded k/n, rest in first-stage order"
```

*Intention, line by line:*

- **First-stage order:** the budget is spent on the most promising candidates first.
- **"No new call starts":** a call in flight finishes, so the overrun is at most one call.
- **Ungraded keep their first-stage order, score and stage:** a cut-off after k grades is
  exactly rerank depth k, and a trace shows they were never graded.
- **The clock is injected**, so unit tests are deterministic.
- **Degradation is recorded per question:** `RetrievalResult.degraded` →
  `QueryTrace.degraded`. This also fixed an old gap: CLAUDE.md §9 asks for "reranker down →
  fused ranking, **flagged in the trace**", and failures were only *counted* before.

### The ranking policy was chosen by simulation

| k graded | A: ungraded after all graded | B: ungraded before grades 0/1 | C: grades 2-3, then first-stage order |
| --- | --- | --- | --- |
| 5 | 0.583 | **0.635** | 0.583 |
| 10 | **0.682** | 0.651 | **0.682** |
| 15 | **0.708** | 0.651 | **0.708** |

B only helps at extreme cut-offs. From k = 10, first-stage ranks 11-20 hold evidence *less*
often than candidates the grader marked 0 or 1: a small model's low grades are a weak
signal, and first-stage position is a better one. **A was chosen:** it ties C and is the
simplest to explain.

### Result

| | no budget | **13 s budget** |
| --- | --- | --- |
| retrieve p50 / p95 (cold) | 28.9 / 32.4 s | **16.3 / 17.7 s** |
| slowest rerank | — | 14.6 s (overrun < 1 call) |
| grades per question | 20 | 9-16 (adapts to passage length) |
| recall@5 | 0.740 | 0.688 (tie with depth 10) |
| questions flagged `degraded` | 0 | 96/96 |

A budget *is* the depth curve, expressed in seconds, with one advantage a fixed depth
lacks: on a slower or loaded server it grades fewer and still keeps the promise. The
"balanced" profile is `planner-llm-rerank-llm-budget13.yaml`.

---

## 6. Step 3: keeping table rows together (backlog 3.1, HLD 021)

### Problem the data showed

The baseline parser puts **every table cell on its own line**:

```text
Purchases of property, plant and equipment (PP&E)
 
 
(1,577)
 
 
(1,373)
```

### Code and intention

- **`PypdfLayoutParser`** uses pypdf's own layout mode (no new dependency), and
  `normalize_layout_text()` compacts it:

  ```text
  (Millions) | 2018 | 2017 | 2016
  Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)
  ```

  - column gaps of 3+ spaces become ` | `;
  - `$` and `%` stick to their numbers, but `%` never joins a following word (`2017 | %
    change`);
  - line-end hyphens are **kept**: re-joining them produced "localcurrency".
- **"Never lose text."** The first parse audit found AMEX's 10-K cover page empty. Layout
  mode lost words on 31 of 11,948 pages, so a page whose layout text has fewer than 80% of
  plain extraction's words falls back to plain text. The fallbacks are counted.
- **Artifacts never mix.** Parsed documents go to `parsed_dir_for(parser)`, and the parser
  joins the chunk fingerprint, but only for non-baseline parsers. Every existing index and
  cache kept its paths and IDs; this was checked before any run.

### Result

| | pypdf | layout |
| --- | --- | --- |
| parse audit pass rate | 96.8% | **97.4%** |
| evidence in one chunk | **72.0%** | 67.7% |
| recall@5 default / best | 0.484 / 0.740 | 0.490 / 0.745 (ties) |
| generation numeric (llama3.1:8b) | **10/32** | 8/31 (5 worse / 3 better: n.s.) |

**Reading the flipped answers explained the tie.** The model read the right figure (Costco
total assets $59,268) and then reasoned wrongly: it subtracted the prior year, unasked. In
another, it used net income where operating income was needed. The bottleneck is the 8B
model's reasoning, not table reading. Opt-in. Step 2 (column headers on every row) is
deferred until a stronger generator can test it.

---

## 7. Step 4: chunk size and overlap (guide Part 15, HLD 022)

### Code and intention: compare at equal cost

```python
def recall_at_token_budget(
    ranked: Sequence[str], token_counts: Sequence[int], qrels: Qrels, budget: int
) -> float:
```

It walks the ranking and keeps chunks while their prompt tokens (the compacted header +
text, as the generator sees them) fit the budget. A chunk that doesn't fit *ends* the
prefix rather than being skipped, because the generator reads sources in rank order. It
has hand-computed tests, as for every metric (CLAUDE.md §6).

### Result: three views and one decider

| Chunks | recall@5 | recall @2,000 tok (page) | evidence text @2,000 tok | truncated by bge |
| --- | --- | --- | --- | --- |
| 200 / 0 | 0.469 | **0.656 (+0.172 \*)** | 0.344 | 69 |
| **400 / 0** | 0.484 | 0.484 | 0.312 | 93 |
| 400 / 40 | 0.505 | 0.516 | 0.365 | 93 |
| 800 / 0 | 0.516 | 0.443 | 0.385 | **8,313 (45%)** |

**Answers at the same budget:** 400 × 5 sources gave **13/30** numeric correct; 200 × 9
sources gave **5/31** (8 vs 1 paired flips, p ≈ 0.04).

- **At an equal chunk count, size doesn't matter:** the guide's plot is flat.
- **At an equal token budget, 200 looks much better, but only on page labels.** Many small
  chunks touch many pages. On the evidence *text* the gain disappears.
- **The answers decided it:** fragments hurt. A table cut into 200-token pieces is a row
  here and a header there.
- **800 can't be tested fairly with bge-small:** its 512-token window cuts off 45% of those
  chunks.

**400 / 0 kept.**

---

## 8. Step 5: parent-child retrieval (backlog 3.7, HLD 023)

### Problem

Step 4 measured both halves of the trade-off: small chunks find pages, big chunks make
answers. Parent-child (Parts 4 and 12) aims for both.

### Code and intention

```python
def retrieve(self, query: Query, k: int) -> RetrievalResult:
    result = self._children.retrieve(query, k * self._fanout)
    parents: list[ScoredChunk] = []
    seen: set[str] = set()
    for child in result.candidates:
        parent = self._parent_of.get(child.chunk.chunk_id)
        if parent is None or parent.chunk_id in seen:
            continue
        seen.add(parent.chunk_id)
        parents.append(
            ScoredChunk(chunk=parent, score=child.score, rank=len(parents) + 1, stage=STAGE)
        )
        if len(parents) == k:
            break
    return result.model_copy(
        update={
            "candidates": tuple(parents),
            "stage_candidates": {**result.stage_candidates, CHILD_STAGE: result.candidates},
        }
    )
```

*Intention, line by line:*

- **`k * fanout` children:** several children share one parent, so ask for more.
- **The parent takes its best child's score and position**, and duplicates are skipped.
- **The children stay in `stage_candidates`**, so a trace shows which small chunk found the
  parent.
- **The parents are exactly the default 400-token chunks, with the same IDs.** Recall@5,
  the LLM grade cache and the prompt size stay directly comparable, and only *how parents
  are found* changes.
- **`map_children_to_parents`** assigns each child to the same-document parent with the
  largest character overlap. Both chunkers are page-bounded and read the same text, so
  offsets line up.

### Result

| | default | default + pc | best | best + pc |
| --- | --- | --- | --- | --- |
| recall@5 | 0.484 | 0.479 | 0.740 | **0.781** (+0.042, CI [−0.026, +0.109]) |
| hit@5 | 0.521 | 0.510 | 0.781 | **0.844** (+0.062, 9 better / 3 worse) |
| pool recall | 0.760 | **0.885** | 0.976 | 0.983 |

- **Children find more of the right parents but don't rank them better.** The pool grows
  by 0.125 and the top 5 doesn't change, until the LLM reranker uses the bigger pool.
- **It's the most promising result since the reranker, but not significant at n=96.** So
  it's opt-in, and **it wasn't run on test**: doing so to decide would mean choosing by
  test.

### Not built: LLM-written chunk context

Contextual retrieval needs an LLM call per chunk, given the document. Documents are a median
107k tokens, beyond llama's window, so the whole-document method is impossible locally. A
page-window variant is ~59M input tokens, **~47 hours** on this Mac. It needs an approved
paid run.

---

## 9. How the evaluation evolved

| Before | After |
| --- | --- |
| latency numbers from cached runs | cold vs warm, per-stage p50/p95, measured awake on AC, baseline repeated |
| reranker fallback only counted | every fallback and cut-off **flagged per question** (`degraded`) |
| recall@k only | + **recall at an equal token budget** |
| page-level recall decides | page → evidence text → answers, before any ingestion decision |
| one parser, one artifact folder | per-parser artifacts and fingerprints; old caches stay valid |
| expensive sweeps re-run the model | exact simulation from cached grades where possible |

---

## 10. Configs and how to reproduce

| Config | What it is | Status |
| --- | --- | --- |
| `planner-llm-rerank-llm-budget13.yaml` | best + 13 s rerank budget | **balanced profile** |
| `planner-rules-layout.yaml`, `planner-llm-rerank-llm-layout.yaml`, `generation-ollama-best-layout.yaml` | layout parser | opt-in |
| `chunks-{200,400,800}-o{0,10}.yaml`, `generation-ollama-chunks200.yaml` | chunk sweep | records |
| `planner-rules-pc.yaml`, `planner-llm-rerank-llm-pc.yaml` | parent-child | opt-in (needs the `chunks-200-o0` index) |

```bash
# cold latency (keep the Mac awake and the lid open)
caffeinate -dims uv run python scripts/evaluate_retrieval.py --config <config with llm cache: false> --limit 24
# balanced profile
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm-rerank-llm-budget13.yaml
# layout parser: build its artifacts once
make data CONFIG=configs/planner-rules-layout.yaml
# parent-child: build the child index once, then evaluate
make data CONFIG=configs/chunks-200-o0.yaml
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm-rerank-llm-pc.yaml
# equal-budget comparison of finished runs
uv run python scripts/budget_recall.py runs/<a> runs/<b> --baseline runs/<a>
```

---

## 11. Reading order for this code

1. `retrieval/llm_reranker.py`: `LLMPointwiseReranker.rerank` and `_grades`. The time
   budget, the cache and concurrency are all in one loop.
2. `retrieval/base.py` (`degraded`, `last_degradation`), then `retrieval/reranking.py`.
3. `ingestion/pdf_parser.py`: `normalize_layout_text`, then `PypdfLayoutParser._page_text`
   (the fallback).
4. `config/settings.py: parsed_dir_for` and `ingestion/chunkers.py: fingerprint` (the
   `source` key): how artifacts stay separate.
5. `evaluation/retrieval_metrics.py: recall_at_token_budget` with its test, then
   `scripts/budget_recall.py`.
6. `retrieval/parent_child.py`, then `app.py: load_children`.

Design records: HLD 019 (latency), 020 (budget), 021 (layout parser), 022 (chunk sweep),
023 (parent-child), each with results, a decision and lessons.

---

## 12. Lessons of this stretch

1. **Measure first:** 90% of the latency was one stage, and the plan followed from that.
2. **Small probes mislead.** Concurrency's 1.93× disappeared in the full pipeline.
3. **Output-format "overhead" can be thinking room.** `{"grade": ` and `working` both carry
   meaning for a small model.
4. **A bound is a feature even when it isn't a speed-up.** The time budget turned a
   trade-off curve into a latency promise, with every cut-off visible.
5. **Labels can reward the wrong thing.** Page-level recall favoured fragments that
   produced worse answers.
6. **Read the failures:** the layout parser's flipped answers pointed at the generator, not
   the parser.
7. **"Promising" isn't "significant".** Three changes landed between +0.02 and +0.06 with
   intervals that include 0. 96 dev questions can't resolve them.

---

## 13. Where things stand, and what's next

- **The bottleneck is the generator** (llama3.1:8b): evidence reaches the prompt, but
  reasoning fails. The stronger-generator comparison (HLD 018) is parked until a gateway
  route without reply filtering exists.
- **Statistical power:** grow the evaluation set before more fine-grained ablations.
  Parent-child, 10% overlap and the layout parser all await it.
- **Remaining finish-line items:**
  - 4.3: an automatic evaluation check in CI;
  - 4.4: the BEIR cross-check;
  - 2.2: judge calibration (~30 labels);
  - 2.3 v2: near-miss unanswerable traps;
  - a final ablation write-up across all phases.
