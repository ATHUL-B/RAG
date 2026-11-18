# Phase 3 walkthrough: improving retrieval one measured change at a time

Phase 1 (`phase-1-baseline.md`) built a naive pipeline and the equipment to measure it.
Phase 3 used that equipment: **find the biggest failure in the traces, change one thing to
address it, measure, keep or reject, repeat.** This document tells that story step by step
and explains what each new piece of code is for, so you can read the code with its
intention in mind.

Read Phase 1's walkthrough first; this one assumes its vocabulary (traces, relevance units,
the debugging ladder, ablation configs). Study-guide references look like **(Part 8)**.

---

## 1. TL;DR

| # | Step | Backlog | Question it answered | recall@5 | Kept? |
| --- | --- | --- | --- | --- | --- |
| 0 | Phase 1 baseline: dense top-5 | 1.7 | - | 0.273 | - |
| 1 | **Metadata filters** from the question | 3.3 | Can we stop retrieving the wrong filing? | **0.377** | yes (significant) |
| 1b | … with a ±1 year window | 3.3 | Do neighbouring years help forward-looking questions? | 0.337 | no |
| 2 | **BM25** alone | 3.4 | Do exact terms beat embeddings here? | 0.293 | no (significantly worse) |
| 2b | **Hybrid** dense + BM25 with RRF | 3.4 | Do the two lists complement each other? | 0.360 | tie; pool better |
| 3 | **Cross-encoder reranker** (MiniLM, bge) | 3.5 | Can a reranker pick the evidence out of the pool? | 0.333-0.367 | no (ties / worse) |
| 4 | **Page-aware chunks** | 3.2 | Do chunks that mix statements hurt? | 0.390 | yes (tie, but chunk audit 62% → 72%) |
| 4b | + **Context headers** | 3.2/3.7 | Does "which statement, which year" help search? | **0.427** | yes (**new default**) |
| 4c | + hybrid | 3.4 | Does hybrid help once chunks have context? | 0.437 | optional (tie) |
| 4d | + reranker | 3.5 | Does context rescue the reranker? | 0.343 | no (significantly worse) |

Baseline → best is **+0.163 recall@5, significant** (95% CI [+0.087, +0.243]).
But after step 1, **no single step is significant on its own**: 150 questions can't
resolve differences of about 0.03. That is the main lesson about evaluation.

Current default: `configs/pages-context.yaml` (filters + page-aware contextual chunks +
dense retrieval).

---

## 2. The method: how each step was chosen

Every step followed the same loop (Part 13, "ablation ladder"):

1. **Look at the traces of the current best run** and find the largest group of failures.
   The question is not "what technique is popular?" but "*where* is evidence being lost?".
2. **Write an HLD** with the hypothesis and the metric that should move.
3. **Implement it behind the existing interfaces**, so the change is one new class plus one
   config value.
4. **Run an ablation config that differs in exactly one setting** (a test enforces this).
5. **Compare per question with a confidence interval** (`scripts/compare_runs.py`, added in
   step 2), look at segments, and **read real examples** before believing any number.
6. **Keep, reject or keep-as-optional**, and record why in `experiments/LOG.md` and in the
   HLD's *Outcome* section.

The order was 3.3 → 3.4 → 3.5 → 3.2, not the backlog numbering, because each result
pointed to the next bottleneck:

```
baseline: 40.7% of top-5 slots = same company, OTHER filing   → filters (3.3)
filters:  66.7% of top-5 slots = right filing, WRONG page      → keyword search (3.4)
hybrid:   evidence is in the top 50 but not the top 5          → reranker (3.5)
reranker: evidence chunks are header-less table fragments      → structure + context (3.2)
```

---

## 3. What changed in the code, at a glance

### 3.1 New and changed modules

```
src/rag/
  domain/models.py         + Filters / FilterValue / matches()      (step 1)
                           + Chunk.context, Chunk.indexed_text       (step 4)
                           + Document.page_spans()                   (step 4)
  retrieval/
    base.py                ~ Retriever.retrieve() -> RetrievalResult (step 1, grown in 2-3)
                           + FilterExtractor interface               (step 1)
    filters.py             + CatalogFilterExtractor                  (step 1)
    filtered.py            + FilteredRetriever (filters + relaxation) (step 1)
    keyword.py             + KeywordRetriever                        (step 2)
    fusion.py              + ReciprocalRankFusion                    (step 2)
    hybrid.py              + HybridRetriever                         (step 2)
    rerankers.py           + CrossEncoderReranker, PairScorer        (step 3)
    reranking.py           + RerankingRetriever                      (step 3)
    factory.py             + build_retriever / build_bm25_index / build_reranker
  indexing/
    text_analysis.py       + analyze() for keyword search            (step 2)
    keyword_index.py       + BM25Index                               (step 2)
    vector_store.py        ~ uses domain.matches for filters         (step 1)
    index_stage.py         ~ embeds Chunk.indexed_text               (step 4)
  ingestion/
    chunkers.py            ~ RecursiveChunker(respect_pages=...)     (step 4)
    context.py             + page_title_block, HeaderContextualizer, ContextualChunker (step 4)
    factory.py             ~ builds the contextual chunker from config
  datasets/financebench.py + COMPANY_ALIASES                         (step 1)
  evaluation/
    retrieval_runner.py    ~ filtered_out status, filter_ok, stage_ranks, pool_recall,
                             per-stage latency
    compare.py             + paired bootstrap run comparison         (step 2)
  config/settings.py       + RetrievalConfig.{mode, candidate_depth, bm25, fusion,
                             rerank, filters}; ChunkingConfig.{respect_pages, context}
scripts/compare_runs.py    + per-question comparison of two runs
configs/*.yaml             + one file per ablation (11 new)
```

### 3.2 How retrievers compose now

Phase 1 had one retriever. Phase 3 builds a **stack of decorators**, each a `Retriever`
wrapping another `Retriever`. `retrieval/factory.py: build_retriever` assembles it from
config:

```
FilteredRetriever                 extracts company/year, relaxes if nothing is found
  └─ RerankingRetriever           (optional) asks for 50, reranks, keeps k
       └─ HybridRetriever         (optional) runs both with the same filters, fuses with RRF
            ├─ DenseRetriever     embeddings + ExactVectorStore
            └─ KeywordRetriever   BM25Index
```

Why decorators? Each layer has one job and the same interface, so any layer can be turned
on or off from config, and tests can test each layer with fake inner retrievers. This is the
"composition over inheritance" rule from `CLAUDE.md` doing real work.

### 3.3 The one interface change: `RetrievalResult`

In Phase 1, `Retriever.retrieve()` returned a bare `list[ScoredChunk]`. That was not enough
once layers started making decisions the evaluation needs to see, so it now returns:

```python
class RetrievalResult(BaseModel):
    candidates: tuple[ScoredChunk, ...]  # the final ranking
    applied_filters: Filters  # step 1: what the search was limited to
    relaxed: tuple[str, ...]  # step 1: filters dropped, in order
    stage_candidates: dict[
        str, tuple[ScoredChunk, ...]
    ]  # step 2: each inner list ("dense", "bm25", "first_stage")
    timings_ms: dict[str, float]  # step 3: per-stage latency ("first_stage", "rerank")
```

**Intention:** a retriever's result explains itself. The runner copies these into every
trace, so "why was this question missed?" can be answered from the trace alone.

---

## 4. Step 1: metadata filters from the question (backlog 3.3, HLD 006)

### Problem the traces showed

In the baseline, **40.7% of all top-5 slots** were other filings of the *same company*.
"3M FY2018 capex" retrieved 3M's 2023 10-Q and 2022 10-K first: annual reports reuse
near-identical wording every year, so embeddings can't tell the years apart. This is the
guide's version trap (Part 8) on real data.

### What the data said before designing

- 25 of 150 questions don't contain the catalog company name: they say `JnJ`, `JPM`,
  `AMEX`, `Coca Cola`, `MGM` → an **alias table** is needed.
- 128 questions mention the gold filing's year, 19 mention none, 3 mismatch. The mismatches
  are forward-looking ("forecasting for FY2023", answered by the 2022 10-K) and ranges.

### Code and intention

**`domain/models.py`: `Filters` and `matches()`**

```python
FilterValue = MetadataValue | tuple[MetadataValue, ...]  # scalar = equals, tuple = any of
Filters = dict[str, FilterValue]


def matches(metadata: Metadata, filters: Filters) -> bool: ...
```

*Intention:* one definition of "does this chunk pass the filter", used by the vector store,
the BM25 index **and** the evaluator. If they had separate implementations, the evaluator
could disagree with the search about what was allowed.

**`retrieval/filters.py`: `CatalogFilterExtractor`**

- `normalize()` lowercases and turns punctuation into spaces, so `Coca-Cola's`,
  `coca cola` and `COCA-COLA` all match the alias `coca cola`; aliases match whole words only
  (`AMDOCS` is not `AMD`).
- If aliases of **two different companies** match, it returns *no* company filter.
  *Intention:* a wrong hard filter hides the answer completely, which is worse than not
  filtering at all.
- Years: a regex for 1990-2049 that rejects numbers like `12,019`, `2019.5`, `$2,019,000`.
  A bug was found here by measuring: the first version also rejected years followed by a
  comma, which skipped prose like "FY2023, FY2022 and FY2021". Measuring extraction
  accuracy against the gold filings (company right on 146/150, never wrong) caught it.
- `year_window` widens each year by ±N (an ablation knob, see below).

**`datasets/financebench.py`: `COMPANY_ALIASES`**: canonical names, common names and
tickers, written from general knowledge. In production this would be a company master
table. Caveat (also in the HLD): a few aliases were noticed while reading these same
questions, so extraction accuracy is slightly optimistic.

**`retrieval/filtered.py`: `FilteredRetriever`**

```python
filters = {**extracted, **query.filters}       # explicit filters win
while True:
    result = inner.retrieve(query with filters, k)
    if enough results or nothing left to relax: return result (+ which keys were relaxed)
    drop the next key in relax_order            # "year" first, then "company"
```

*Intention:* stepwise relaxation (Part 8). A company with no filing for the asked year
(Boeing "FY2023") would otherwise return nothing; relaxing the year turned it into a hit.
Explicit filters, which in a real system would come from a user's account or permissions,
are never relaxed.

### Evaluation additions

- `filter_ok` metric: did the applied filters let at least one evidence document through?
- New ladder status `filtered_out`: the *filter itself* excluded the evidence. This splits
  rung 2 into "the filter was wrong" vs "ranking was wrong".

### Result and lesson

| | Baseline | Filters (±0) | Filters (±1) |
| --- | --- | --- | --- |
| recall@5 | 0.273 | **0.377** | 0.337 |
| Same company, other filing (top-5 slots) | 40.7% | **18.8%** | 35.3% |
| Other company | 22.1% | 2.9% | 2.9% |
| Right filing, wrong page | 29.9% | **66.7%** | 52.3% |

- The largest and only clearly significant single gain of Phase 3 (+0.103, CI [+0.057,
  +0.153], 21 better / 1 worse).
- The ±1 window fixes 2 forward-looking questions but readmits neighbouring years, which is
  the trap again. Rejected.
- **The bottleneck moved:** two-thirds of the top-5 slots are now the right filing but the
  wrong page. That pointed to step 2.

---

## 5. Step 2: BM25 and hybrid search (backlog 3.4, HLD 007)

### Problem

Within the right filing, the right page was rarely first. Financial questions are full of
exact line-item names ("Purchases of property, plant and equipment") and numbers, which
embeddings blur and keyword search matches exactly (Part 9).

### Code and intention

**`indexing/text_analysis.py`: `analyze()`**: lowercase, `1,577` → `1577` (so a number
matches however it is written), keep decimals (`7.9`), remove a small stop-word list.
*Intention:* "the analyzer matters" (Part 9). Documents and queries must go through the
**same** function or their terms never meet.

**`indexing/keyword_index.py`: `BM25Index`**, written out instead of imported:

```python
IDF(t)      = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)
score(D, Q) = Σ_t IDF(t) · f(t,D)(k1+1) / (f(t,D) + k1(1 - b + b·|D|/avgdl))
```

- An inverted index: term → NumPy arrays of (chunk positions, term frequencies), so a query
  only touches chunks containing its terms.
- The same `matches()` pre-filter as the vector store ("apply exactly the same filter to
  both", Part 8).
- Tests reproduce the guide's numbers: IDF 3.22 / 8.11 and the ×1.375 saturation factor.

*Intention:* make the formula visible and testable; a library would hide the learning.

**`retrieval/fusion.py`: `ReciprocalRankFusion`**

```python
RRF(d) = Σ_lists  weight_list / (k + rank_list(d))      # k = 60
```

Only ranks are used, because cosine similarities (0-1) and BM25 scores (unbounded) can't
be added. Ties break by best individual rank, then first appearance, so output is
deterministic. Tests reproduce the guide's worked example (order B, P, K, Q, A, R;
B = 0.750 and P = 0.700 with k = 1).

**`retrieval/hybrid.py`: `HybridRetriever`** takes **named** retrievers
(`{"dense": …, "bm25": …}`), gives each the same query and filters with
`candidate_depth` = 50, and fuses. Each inner list is kept in `stage_candidates`.
*Intention:* names make traces readable and make RRF weights configurable per retriever.

**`evaluation/compare.py` + `scripts/compare_runs.py`: the paired bootstrap.** Added in
this step because the first hybrid result was a small difference and the question "is this
real?" was unavoidable:

```
diffs = per-question (B - A)
resample diffs with replacement 5,000 times → distribution of the mean difference
95% CI = 2.5th and 97.5th percentiles;  CI excludes 0  →  likely a real difference
```

*Intention:* stop reading noise as progress. From here on every step was judged this way.

**Runner additions:** `stage_ranks` (rank of the first evidence chunk in each inner list:
*which* retriever found it) and `pool_recall` (evidence anywhere in the candidate pool:
the ceiling a reranker could reach, Part 13).

### Result and lesson

| (all with filters) | recall@5 | hit@50 |
| --- | --- | --- |
| dense | **0.377** | 0.820 |
| BM25 only | 0.293 (significantly worse) | 0.733 |
| hybrid | 0.360 (tie: 10 better / 13 worse) | **0.887** (union) |

- The guide's expectation "hybrid beats dense" did **not** hold at top 5 on this data.
- BM25-only loses because the questions are long and full of instruction boilerplate
  ("…relying on the details shown in the cash flow statement"), which matches generic
  statement pages.
- But the lists are **complementary**: 10 questions are found *only* by BM25, and the union
  raises hit@50 from 0.820 to 0.887. Equal-weight RRF rewards agreement, so single-list
  finds rarely reach the top 5.
- RRF weights and BM25 parameters were deliberately **not tuned**: tuning on the same 150
  questions we score on would overfit (backlog: dev/test split first).
- The pool contained the evidence 85.6% of the time, so the natural next step was a
  reranker to pick it out.

---

## 6. Step 3: cross-encoder reranking (backlog 3.5, HLD 008, ADR 0004)

### Problem

The evidence is usually *in* the top 50 but not in the top 5. A cross-encoder reads the
question and a chunk **together** in one pass and should judge "does this answer it?" far
better than comparing two separately-made vectors (Part 10).

### Code and intention

**`retrieval/rerankers.py`**

- `PairScorer` protocol: `score(pairs) -> list[float]`. *Intention:* a seam, so tests use a
  word-overlap fake and the real model is only loaded in `slow` tests.
- `SentenceTransformersPairScorer`: lazy-loaded `CrossEncoder` with a pinned revision and
  `device: auto`.
- `CrossEncoderReranker.rerank()`: score every pair, stable sort, keep `top_n`, tag
  `stage="rerank"`. Raises if the scorer returns the wrong number of scores.

**`retrieval/reranking.py`: `RerankingRetriever`**: ask the inner retriever for `depth`
(50) candidates, rerank, return `k`. It records the pre-rerank list as `first_stage` and
times both stages. *Intention:* reranking is the first stage that is *expensive*, so its
cost has to be visible per question (Part 14 latency budget).

**Runner:** per-stage latency percentiles (`stage_latency_ms`), printed by the eval script.

Two models were compared (ADR 0004): `ms-marco-MiniLM-L-6-v2` (23M parameters, ~325 ms
per 50 pairs) and `bge-reranker-base` (278M, ~1.7 s).

### Result and lesson

| (all with filters) | recall@5 | nDCG@5 | recall@10 | latency p50 |
| --- | --- | --- | --- | --- |
| hybrid | 0.360 | 0.295 | 0.440 | 12 ms |
| hybrid + MiniLM | 0.333 | 0.252 (significantly worse) | 0.490 | 232 ms |
| hybrid + bge | 0.340 | 0.245 | 0.489 | 1,558 ms |
| dense + bge | 0.367 | 0.260 | 0.487 | 1,577 ms |

No gain at top 5, for 220 ms to 1.5 s more per question. Instead of accepting that, one
question was traced (3M FY2018 capex):

- The reranker's #1 was the **segment table on page 38**, which contains "$1,577": a
  *correct* answer on a page FinanceBench didn't label. **Labels are incomplete.**
- The labelled cash-flow page had been cut into **header-less fragments**
  ("Acquisitions, net of cash acquired 13 (2,023) …") that the reranker scored 0.004-0.065.
  Nothing in the chunk says which statement or which year.

An aggregate check (does the answer's number appear in the top 5?) showed the reranker was
not systematically finding answers elsewhere, so the main diagnosis was **chunk context,
not ranking**. The reranker shipped tested but disabled by default, with a hypothesis to
re-test after step 4.

---

## 7. Step 4: page-aware chunks and context headers (backlog 3.2 / 3.7, HLD 009)

### Problem

Chunks lacked the context needed to judge them, and recursive chunks freely crossed page
boundaries, mixing e.g. the end of the equity statement with the start of the cash-flow
statement.

### What the data said before designing

- pypdf keeps no fonts, and italic fragments ("Is a Service Contract") look like headings,
  so generic heading detection was unreliable.
- In 10-K filings the **page** is the structural unit that matters: each statement starts
  on its own page with a **title block** (`3M Company and Subsidiaries / Consolidated
  Statement of Cash Flows / Years ended December 31 / (Millions) / 2018 / 2017 / 2016`).
  A prototype found a title on 158 of 189 evidence pages.

### Code and intention

**`Chunk.context` and `Chunk.indexed_text`** (`domain/models.py`)

```python
context: str = ""  # e.g. "3M | 10k | 2018 | ... Consolidated Statement of Cash Flows ..."


@property
def indexed_text(self) -> str:  # what embedders, BM25 and rerankers read
    return f"{self.context}\n\n{self.text}" if self.context else self.text
```

*Intention:* separate **what we search** from **what we cite**. `text` must stay exactly
`Document.text[char_start:char_end]` so offsets, audits and (in Phase 2) citations remain
exact; the context only affects search. `IndexStage` (embedding and its truncation
check), `BM25Index.add` and `CrossEncoderReranker` were switched to read `indexed_text`.

**`RecursiveChunker(respect_pages=True)`** with `Document.page_spans()`: the split-and-merge
runs inside each page, so no chunk crosses a page.
*Intention, and a compatibility detail:* the `respect_pages` key is added to the
fingerprint **only when true**, so every existing chunk set and index kept its ID and none
had to be rebuilt.

**`ingestion/context.py`**

- `page_title_block()` takes the short leading lines of a page until the first long
  (prose) line, up to 8 lines; drops "Table of Contents" even when PDF extraction broke it
  into "T / able of / Contents", drops bare page numbers (but keeps years), and re-joins
  split words ("Flow" + "s" → "Flows"). Prose pages return "".
- `ChunkContextualizer` interface + `HeaderContextualizer`: `company | doc_type | year |
  <page title block>`.
- `ContextualChunker`: a **decorator** around any chunker. Same boundaries, plus context, and
  it re-derives chunk IDs and the fingerprint from both parts. *Intention:* contextual chunks
  get their own chunk directory and index, so they can never be mixed with plain ones.

The capex chunk after this step:

```
context: 3M | 10k | 2018 | 3M Company and Subsidiaries / Consolidated Statement of Cash
         Flows / Years ended December 31 / (Millions) / 2018 / 2017 / 2016 / ...
text:    Cash Flows from Investing Activities Purchases of property, plant and equipment
         (PP&E) (1,577) (1,373) (1,420) ...
```

### Result and lesson

| (all with filters) | recall@5 | nDCG@5 | recall@10 | evidence in one chunk |
| --- | --- | --- | --- | --- |
| dense, recursive (before) | 0.377 | 0.292 | 0.468 | 62.4% |
| + page boundaries | 0.390 | 0.309 | 0.493 | **72.0%** |
| + context headers (**default**) | **0.427** | **0.339** | 0.547 | 72.0% |
| + hybrid | 0.437 | 0.321 | 0.563 | |
| + bge reranker | 0.343 (significantly worse) | 0.252 | 0.503 | |

- Each step is a tie on its own; the trend is consistently positive and the whole ladder
  from the baseline is significant.
- Segments move in opposite directions: pages help financial-statement questions
  (`metrics-generated` 0.48 → 0.67); context helps `domain-relevant` (0.18 → 0.36) but pulls
  `metrics-generated` back to 0.50. An untested guess: one shared page header makes every
  chunk on a statement page look alike.
- **The step 3 hypothesis was rejected:** the reranker got significantly worse *with*
  context. Writing the hypothesis down beforehand is what made this a clear finding rather
  than something to explain away.

---

## 8. How the evaluation itself evolved

| Added | Where | Why it was needed |
| --- | --- | --- |
| `filter_ok`, `filtered_out` | `retrieval_runner.py` | Tell a wrong filter apart from wrong ranking (step 1) |
| `applied_filters`, `relaxed` in traces | `retrieval_runner.py` | See exactly what each search was limited to |
| `stage_ranks` | `retrieval_runner.py` | Which retriever found the evidence, at which rank (step 2) |
| `pool_recall` | `retrieval_runner.py` | The ceiling for any reranker: it can only reorder the pool (step 2) |
| Paired bootstrap | `evaluation/compare.py`, `scripts/compare_runs.py` | Separate real differences from noise (step 2 onward) |
| Per-stage latency | `RetrievalResult.timings_ms`, `stage_latency_ms` | Make expensive stages (rerank) visible (step 3) |
| Config tests per ablation | `tests/config/test_loader.py` | Guarantee each ablation changes exactly one section |

Lessons about evaluation, which matter as much as the retrieval results:

1. **Measure the data before designing** (alias needs, year patterns, title blocks):
   every design in Phase 3 was shaped by a quick look at real questions or pages.
2. **Read examples before believing a number.** The reranker "failure" contained a
   correct answer on an unlabelled page; the BM25 failure was question boilerplate.
3. **Small evaluation sets hide small effects.** 150 questions resolve the filter step
   (+0.10) but not +0.01-0.04 steps.
4. **Don't tune on the test set.** RRF weights, BM25 parameters and title-block limits were
   left at defaults on purpose.

---

## 9. Configs and how to reproduce

Every ablation config differs from its parent in exactly one section (enforced by tests):

```
baseline.yaml
└─ filters.yaml                         + retrieval.filters             (step 1)
   ├─ filters-window1.yaml              year_window: 1                  (step 1b)
   ├─ bm25.yaml                         mode: bm25                      (step 2)
   ├─ hybrid.yaml                       mode: hybrid                    (step 2b)
   │  ├─ rerank-minilm.yaml             + rerank (MiniLM)               (step 3)
   │  └─ rerank-bge.yaml                + rerank (bge)                  (step 3)
   ├─ dense-rerank-bge.yaml             + rerank (bge) on dense         (step 3)
   └─ pages.yaml                        chunking.respect_pages          (step 4)
      └─ pages-context.yaml  ← DEFAULT  + chunking.context              (step 4b)
         ├─ pages-context-hybrid.yaml   + mode: hybrid                  (step 4c)
         └─ pages-context-rerank.yaml   + rerank (bge)                  (step 4d)
```

A config that changes **chunking** needs its own chunks and index first:

```bash
uv run python scripts/chunk_documents.py --config configs/pages-context.yaml
uv run python scripts/build_index.py     --config configs/pages-context.yaml   # ~4 min
uv run python scripts/evaluate_retrieval.py --config configs/pages-context.yaml
uv run python scripts/compare_runs.py runs/filters-<ts> runs/pages-context-<ts>
```

Configs that only change **retrieval** reuse the index (it is keyed by chunker fingerprint +
embedding model), and the BM25 index is rebuilt in memory in about 1.5 s.

---

## 10. Reading order for the Phase 3 code

1. `retrieval/base.py`: `RetrievalResult` and the interfaces.
2. `domain/models.py`: `Filters`, `matches`, `Chunk.context` / `indexed_text`, `page_spans`.
3. `retrieval/factory.py`: how config becomes the decorator stack (§3.2).
4. Step 1: `retrieval/filters.py` → `retrieval/filtered.py`, with `tests/retrieval/test_filters.py`.
5. Step 2: `indexing/text_analysis.py` → `indexing/keyword_index.py` → `retrieval/fusion.py`
   → `retrieval/hybrid.py`; tests contain the guide's worked numbers.
6. `evaluation/compare.py`: the paired bootstrap, short and worth understanding.
7. Step 3: `retrieval/rerankers.py` → `retrieval/reranking.py`.
8. Step 4: `ingestion/context.py` and the `respect_pages` branch in `ingestion/chunkers.py`.
9. `evaluation/retrieval_runner.py`: how all the diagnostics reach the traces.

Design records: HLD 006-009 (each ends with an *Outcome* section), ADR 0004.

---

## 11. Open questions carried into the backlog

| Question | Why it's open | Backlog note |
| --- | --- | --- |
| Are pages / context / hybrid real gains? | Each step is a tie on 150 questions | Grow the eval set, dev/test split |
| Why do rerankers hurt here? | Even with context they are significantly worse | Truncation, header dominance, web-trained models on tables; try bge-reranker-v2-m3 or an LLM reranker |
| Why does context hurt `metrics-generated`? | Opposite segment effects | Per-chunk context (nearest row/section label), LLM-written context (3.7) |
| Incomplete page labels | Correct answers on unlabelled pages count as misses | Answer-level evaluation in Phase 2 |
| Forward-looking questions ("expected in FY2023") | The rule-based year filter can't map intent to the previous filing | LLM query planner (3.6) |
| Tables as text | Financial statements are flattened into number soup | Table-aware parsing (3.1) |

Phase 2 (generation) comes next: with an LLM answering from the retrieved chunks, the
evaluation can finally judge *answers*, which also addresses the incomplete-label problem.
