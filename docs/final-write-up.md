# Final write-up: what we built, what moved the numbers, and a playbook for any corpus

This closes the learning project. **Part A** is the FinanceBench story: every measured step
from recall@5 0.273 to 0.781, what failed and why, and where the system stands. **Part B**
turns it into a framework you can run on *any* corpus at *any* scale: the order of work,
how to diagnose, which lever to pull for which symptom, and what changes at larger scale.

Every number in Part A comes from `experiments/LOG.md`, with the HLD that explains it.
Part B is the generalisation. Where it goes beyond what we measured, it says so.

---

# Part A: The FinanceBench story

## A1. The system in one paragraph

SEC filing PDFs (84 filings, 12k pages) go through four steps:

1. **Parse** with pypdf, page by page (HLD 002).
2. **Chunk** into 400-token, page-aware chunks with a context header: company, filing,
   year, page title (HLD 003, 009).
3. **Embed** with bge-small-en-v1.5 and search exactly with NumPy (ADR 0003). HNSW is
   available as opt-in (HLD 028).
4. **Retrieve:**
   - company/year **hard filters** taken from the question (HLD 006);
   - a **query planner** (rules by default, LLM optional) that searches several
     rewrites and fuses them with RRF (HLD 011-013);
   - an optional **LLM reranker** with a time budget (HLD 014, 019, 020);
   - optional **parent-child** retrieval (HLD 023).

A local llama3.1:8b answers with citations, shows its working and can refuse (HLD 010,
015, 017). Evaluation and operations around it:

- every run leaves a trace, and a debugging ladder names the failing stage;
- a dev/test split grouped by company, paired bootstrap CIs, and a CI gate (HLD 024);
- the metric code is cross-checked against published BEIR scores (HLD 025).

## A2. The recall ladder: the steps that moved the number

recall@5 means a labelled evidence page was among the top 5 chunks.

| # | Change (one at a time) | recall@5 | Significant? | HLD |
| --- | --- | --- | --- | --- |
| 0 | Baseline: pypdf → recursive 400/0 → bge-small → exact top-5 | 0.273 | — | 002-005 |
| 1 | **Company + year filters from the question** | **0.377** (+0.104) | yes | 006 |
| 2 | Chunks never cross a page | 0.390 | n.s. | 009 |
| 3 | **Context headers** (filing + page title) on the indexed text | 0.427 | n.s. per step | 009 |
| | *(dev/test split introduced; same config on dev: 0.427)* | | | |
| 4 | **Rule-based query planner** (clean boilerplate, FY22 years, forecast → prior filing, abbreviations) | **0.484** dev / 0.481 test | yes, both | 011 |
| 5 | **Local LLM planner** (up to 4 sub-queries, RRF) | 0.583 dev / 0.546 test | dev yes, test n.s. | 012 |
| 6 | **Local LLM reranker** (grade top 20 pointwise) | **0.740** dev / **0.667** test | yes, both | 014 |
| 7 | Parent-child (search 200-token children, return 400-token parents) | 0.781 dev | n.s. | 023 |

Rows 0-3 were measured on all 150 questions, before the split existed. From row 4 on,
every decision was made on **dev (96)** and confirmed on **test (54)** only a few times.

**Shipped profiles:**

| Profile | Config | recall@5 (dev) | Latency |
| --- | --- | --- | --- |
| default | `planner-rules.yaml` | 0.484 | 16 ms |
| balanced | `planner-llm-rerank-llm-budget13.yaml` | 0.688 | p50 16.3 s / p95 17.7 s cold |
| best | `planner-llm-rerank-llm.yaml` | 0.740 | 28.9 s cold, 42 ms warm |

## A3. What did *not* move the number (and why that mattered)

Most experiments ended in "no". Each one still taught something:

| Tried | Result | Lesson | HLD |
| --- | --- | --- | --- |
| Year window ±1 on filters | 0.377 → 0.337 | A looser hard filter admits wrong years; the precise filter wins | 006 |
| BM25 alone / hybrid (RRF) | 0.293 / 0.360 vs dense 0.377 | Top-5 ties, but the **candidate pool** improves (hit@50 0.820 → 0.887) | 007 |
| Cross-encoder rerankers (MiniLM, bge-base) | 0.333-0.340 vs 0.360; over the LLM planner 0.583 → 0.510 | Rerankers trained on web search mis-rank 10-K tables. An LLM grader with a rubric did better | 008 |
| Up-weighting the original query in RRF | no weight helps | Pool recall was 0.976: the evidence was *already in the pool* at rank 3-5, so this was a ranking problem → reranker | 013 |
| Concurrent grading | no gain | Ollama has 1 slot. A probe claimed 1.93×; the full run disproved it | 019 |
| One-token grade instead of JSON | rerank −27% time, recall −0.125 | Removing structure from LLM output changes what it computes | 019 |
| Layout-mode (table-aware) parser | retrieval ties; answers 10/32 → 8/31 | The right number reached the model, which then reasoned wrongly | 021 |
| Chunk size 200 / 800, overlap | ties at page level; 200 **worse** in answers (13/30 → 5/31) | **Page-level recall flattered small chunks**; the evidence-text and answer checks reversed it | 022 |
| Refusal prompt v2 | over-refusal down, but more errors on near-miss traps | 8B model put explanations in the wrong field and looped on the prompt's own words | 027 |
| HNSW instead of exact | identical (filters) / −0.010 n.s. | At 31k vectors with 1.4%-selective filters, exact search is right and fast | 028 |

## A4. Generation: the bottleneck moved

Same llama3.1:8b, two retrieval configs (dev, HLD 015):

| | Default retrieval | Best retrieval |
| --- | --- | --- |
| Evidence in the prompt | 0.52 | **0.79** |
| Answer cites the evidence | 0.22 | 0.42 |
| Numeric answer correct | **13/30** | 10/32 |

Better retrieval reached the prompt but not the answer. "Evidence in the prompt, answer
wrong" (ladder rung 5) became the biggest failure group. The errors were scale
(thousands vs millions), ratio vs percent, and the wrong table. Other generation findings:

- **Integrity checks.** The first full generation run lost 78% of its outputs to schema
  and markup problems. The checks that came from it (HLD 010, 2.1a) mean a broken run
  can no longer look valid.
- **The `working` field.** Asking for structured working before the answer raised
  numeric correctness from 1/8 to 5/8 on the first probe (HLD 015).
- **Refusals.**
  - On the easy traps (v1, 26 questions), refusal recall is 1.00.
  - On near-miss traps (v2, 24 questions) it falls to 0.96 / 0.85. The model
    substitutes the neighbouring period or uses outside knowledge, and **better
    retrieval made it worse**, because more plausible near-miss text reaches the prompt
    (HLD 017, 026).
- **Judges.** qwen2.5:7b agrees with `numeric_match` 90-96%, but only 4/50
  human-labelled calibration items exist. Its scores are **provisional** (HLD 016).

## A5. Production concerns

- **Latency (HLD 019, 020).** 90% of the best profile's 28.9 s is the LLM reranker: 20
  calls of about 1.3 s each. Caches keyed by model + prompt/schema hash make warm
  queries 42 ms. A rerank time budget bounds cold p95 to 17.7 s and flags degraded
  results in the trace.
- **CI gate (HLD 024).** 32 dev questions over 14 filings, with no LLM, run in about 2
  minutes. Company filters make the small corpus return *identical* results to the full
  one. The gate fails if more than 1 question loses recall or the mean falls by more
  than 0.05.
- **Metric code verified (HLD 025).** Our embedder, store and metrics reproduce the
  published MTEB scores. SciFact nDCG@10 is 0.7127 vs 0.7127, and NFCorpus and FiQA
  agree within 0.001.
- **Approximate search (HLD 028).**
  - The from-scratch HNSW agrees with faiss within 0.03. At M=16, ef=64 it reaches
    index recall 0.98 while comparing 2.4% of the vectors.
  - Post-filtering an approximate search loses up to 45% of the filtered top-10.
  - Searching the allowed chunks exactly is both correct and cheapest at our filter
    selectivity.

## A6. Ten lessons, each with its evidence

1. **Verify the measuring stick before the system.** BEIR matched to 4 decimals, so
   every later number could be trusted. The first generation run's 78% loss was a
   *measurement* bug, caught before anyone tuned a prompt on it.
2. **Cheap structure beats model capacity.** Metadata filters (+0.104) and context headers
   cost nothing at query time. Together they gained more than any model swap except the
   LLM reranker.
3. **Split "is it in the pool?" from "is it ranked?".** Pool recall vs recall@k told us
   to stop rewriting queries (pool 0.976) and start reranking (+0.157).
4. **Proxies lie; check one level down.** Page-level recall said 200-token chunks were
   +0.17 at an equal budget. The evidence-text check and the answers said they were
   worse. Every proxy gain needs confirming at the next stage down.
5. **Every gain moves the bottleneck.** After retrieval reached 0.79 evidence-in-prompt,
   the generator became the limit. The debugging ladder tells you where to work next.
6. **A domain-specific judgement beats a generic model.** An 8B LLM grading with a
   rubric beat web-trained cross-encoders on financial tables. Test model families on
   *your* data.
7. **Latency is set by LLM calls inside the loop.** Bound them with time budgets and
   depth, cache them, and record degradation instead of hiding it.
8. **Probes mislead; confirm on full runs.** The 1.93× concurrency speed-up and a
   "<$0.01" cost estimate were both wrong on the full run.
9. **Negative results are results.** Most experiments here were ties or losses. Each
   one closed a door, with the reason written down, and kept the defaults simple.
10. **Small evals need paired statistics.** With 96 dev questions most deltas under
    ~0.05 are noise. Paired bootstrap CIs and a held-out test split stopped us
    shipping noise.

---

# Part B: The playbook, from a new corpus to an optimised system

## B1. The loop

The whole method is one loop, repeated until the target or the budget is reached:

```text
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. Measure    run the eval: per-segment metrics + per-question trace│
    │ 2. Localize   debugging ladder: which stage lost the evidence?      │
    │ 3. Choose     the cheapest lever for that stage (B4), ONE change    │
    │ 4. Test       paired dev comparison; confirm at the next level down │
    │ 5. Decide     adopt / opt-in / reject; write the HLD + LOG row      │
    │ 6. Guard      update the CI gate so the gain can't silently regress │
    └──────────────────────────── back to 1 ─────────────────────────────┘
```

The order of the phases below matters more than any individual technique.

## B2. Phases and exit criteria

### Phase 0: Understand the corpus and the questions (before any code)

| Ask | Why it matters | Here |
| --- | --- | --- |
| What are the documents (PDF, HTML, tables, scans)? How are they structured? | Decides the parser and chunking | Text-layer 10-K PDFs, dense tables |
| What metadata exists (entity, date, version, permissions)? | Hard filters are the cheapest big win | company, year, doc type |
| What will users ask? Types and their share | Segments for reporting and the golden set | single fact, numeric/derived, multi-evidence, unanswerable |
| How big, and how fast does it change? | Index choice, freshness, re-index cost | 31k chunks, static |
| What does a wrong answer cost? | Refusal strictness, the need for citations and judges | Financial: refuse rather than guess |

**Exit:** a list of question segments, metadata fields and constraints (latency, cost,
privacy).

### Phase 1: Build the evaluation first

1. **Golden set:** 50-150 real questions, labelled **at the source** (document + page or
   char span), never by chunk ID, so labels survive re-chunking. Tag each with a segment.
2. **Include traps:** unanswerable questions, *near-miss* ones (an adjacent period, a
   false premise), and version-sensitive ones. They are where systems hallucinate.
3. **Split** into dev and test, grouped by entity (company, product, customer) so test
   entities are unseen. Tune on dev; touch test rarely; never regenerate the split.
4. **Metrics:**

   | Area | Metrics |
   | --- | --- |
   | Retrieval | recall@k, MRR, nDCG, **pool recall** (before rerank) |
   | Index | index recall (approximate vs exact), once an approximate index is used |
   | Generation | evidence-in-prompt, citation validity, correctness, faithfulness, refusal recall / over-refusal |
   | Operations | p50/p95 per stage, cost per query |

5. **Validate the metric code** on a public benchmark with published numbers (BEIR/MTEB)
   before trusting any result.
6. **Paired comparison tooling:** bootstrap CIs, per-question wins/losses.

**Exit:** a baseline number you trust, with a CI.

### Phase 2: The thinnest end-to-end baseline

1. Use sensible defaults, nothing clever:
   - a page-preserving parser;
   - recursive ~400-token chunks;
   - an off-the-shelf embedder;
   - **exact** search;
   - top-5;
   - a cited prompt with a refusal path.
2. Each stage writes an **inspectable artifact** and **deterministic IDs**: chunk IDs
   come from the document, the offsets and a config hash.
3. Run two audits before retrieval is judged:
   - **parse audit:** is the evidence text on the parsed page?
   - **chunk audit:** does the evidence fit in one chunk?

   Here these were 96.8% and 72%. They tell you how much ingestion caps everything
   downstream.
4. Every query writes a **trace**: rewrites, filters, candidates with scores, final
   context, answer, timings.

**Exit:** a baseline LOG row, per segment, and a failure distribution over the ladder.

### Phase 3: Optimise one stage at a time

For each failure group, use the symptom → lever table (B4), cheapest lever first.

| Verdict | When |
| --- | --- |
| **Adopt** | the paired CI excludes 0 on dev, test does not contradict it, and the next stage down agrees |
| **Opt-in** | it is better, but too slow or costly for the default |
| **Reject** | it ties or loses: keep the simpler system |

Write the reason in any case.

### Phase 4: Production hardening

- **Latency budget** per stage, cold and warm, p50 and p95; then cache, cap depth, add
  time budgets and fallbacks.
- Pick the index type by scale (B5), and **measure index recall** when you go
  approximate.
- A **CI regression gate** on a small, deterministic subset.
- Timeouts, retries and degradation flags on every external call.

**Exit:** the SLOs are met, and a regression fails CI.

## B3. The debugging ladder, the core diagnostic

For every wrong answer, find the **first** stage that lost the evidence:

| Rung | Question | Metric that shows it | Stage to fix |
| --- | --- | --- | --- |
| 1 | Is the evidence in the index at all? | parse audit, chunk audit | parsing, chunking |
| 2 | Is it in the first-stage candidates? | **pool recall** | filters, query, embedder, hybrid |
| 3 | Did reranking keep it in the top-k? | recall@k vs pool recall | reranker, depth |
| 4 | Did it reach the prompt? | evidence-in-prompt | context assembly, budget |
| 5 | Was it in the prompt, but the answer wrong? | correctness when in prompt | prompt, generator, verification |

Count failures per rung and work on the biggest **lowest** rung: a fix higher up can't
recover evidence lost below it.

## B4. The lever menu: symptom → what to try, cheapest first

| Symptom | Try first (cheap) | Then (more cost) | Measured here |
| --- | --- | --- | --- |
| **Rung 1:** text missing or garbled after parsing | Better text extraction, normalisation | Layout or table parsing, OCR for scans | Layout parser tied (HLD 021) |
| **Rung 1:** evidence split across chunks | Page/section-aware boundaries | Parent-child; LLM-written chunk context | Page-aware +0.013; parent-child +0.04 on best (n.s.) |
| **Rung 2:** wrong entity or period retrieved | **Metadata hard filters** from the question, with relaxation | An LLM extracts the filters | **+0.104**, the biggest cheap win |
| **Rung 2:** question phrasing ≠ document phrasing | Rule rewrites (abbreviations, boilerplate, fiscal-year forms) | An LLM planner with sub-queries fused by RRF | +0.057, then +0.099 |
| **Rung 2:** exact identifiers or rare terms missed | BM25 hybrid with RRF | A domain or fine-tuned embedder | Pool +0.067, top-5 tie |
| **Rung 2:** chunks of one document look alike | Context headers (title, section, entity) | Bigger or domain embedder | +0.037 |
| **Rung 3:** in the pool, ranked too low | Cross-encoder reranker (test it on *your* data) | LLM pointwise grader with a rubric and a time budget | Cross-encoder −; LLM **+0.157** |
| **Rung 4:** in the top-k, not in the prompt | Raise or reorder the context budget | Dedupe, compress | — |
| **Rung 5:** wrong despite the evidence | Structured working before the answer; unit and scale rules | Stronger generator; answer verification | `working`: 1/8 → 5/8 |
| **Rung 5:** hallucination on near-misses | A refusal path plus a trap set | Evidence/claim verification, stronger model | Prompt rules not enough (HLD 027) |
| **Latency** | Cache (keyed by model + prompt hash), cap depth | Time budgets with degradation; smaller models; skip stages for easy queries | 28.9 s → p95 17.7 s; warm 42 ms |

**Why this order:**

1. **Structure before models.** Filters and headers are free at query time and easy to
   reason about.
2. **Query side before index side.** Changing a query needs no re-index; re-chunking
   or re-embedding means re-ingesting everything.
3. **The pool before the ranking.** A reranker can only reorder what the first stage
   found.
4. **The generator last.** Until evidence reliably reaches the prompt, generator
   changes are measured on noise.

## B5. Scale: what changes as the corpus grows

The method is the same at any scale; the components and their costs change. The
thresholds below come from our measurements: exact search costs about 0.06 ms per 1k
vectors (384 dimensions, single thread), and embedding runs at about 100 chunks/s on a
laptop GPU. Rows beyond ~60k vectors are **extrapolated**, not measured here.

| | Small (≤ 100k chunks) | Medium (100k - 10M) | Large (> 10M) |
| --- | --- | --- | --- |
| **Vector search** | **Exact**: correct, ~6 ms at 100k | HNSW (M 16-32, tune ef on dev for index recall ≥ 0.95); exact for filtered subsets | HNSW/IVF-PQ, sharded; quantised vectors to fit memory |
| **Filters** | Mask then exact | Exact below a threshold (a full-scan fallback); payload index | Partition per tenant/entity; filter-aware indexes |
| **Embedding cost** | Minutes to an hour | ~3 h per 1M chunks on a laptop → GPU or batch API | Days on one machine → a distributed job; the model is an ADR-level choice |
| **Re-index** | Just re-run | Re-index on every chunker or model change: fingerprint the index, never mix models | Blue/green indexes; incremental upserts |
| **Eval** | Full golden set every change | Full set nightly; a **CI gate subset** on each PR | Same, plus online metrics and sampled human review |
| **Latency** | LLM stages dominate | Search plus filter cost becomes visible (our filter mask = 3 ms at 31k) | Retrieval needs its own budget; cache hard; route easy questions past LLM stages |
| **Freshness** | Rebuild | Incremental ingest with deterministic IDs; delete by `doc_id` | Streaming ingest; version-aware retrieval |

Always re-measure index recall at the new size: the `ef` that gave 0.98 at 31k will
give less at 10M.

## B6. Decision rules worth keeping

- **One change per experiment.** A config differs from its base in one field, and a test
  enforces it.
- **Tune on dev, confirm on test, rarely.** A test score never chooses a setting.
- **Adopt only** when the paired CI excludes 0 *and* the next stage down agrees
  (retrieval → evidence text → answer).
- **Ties go to the simpler, cheaper option.** Keep the better-but-slower option as an
  opt-in profile.
- **Latency is a constraint, not an objective.** Set the SLO first, then maximise quality
  within it, as the 13 s rerank budget did.
- **Write down every negative result** with its reason, so nobody runs it again blind.
- **Estimate cost before a paid or long run.** Then check the estimate against the real
  run.

## B7. A one-page checklist for a new corpus

```text
[ ] Phase 0  segments, metadata fields, size/growth, SLOs, cost of a wrong answer
[ ] Phase 1  golden set labelled at source (+ unanswerable + near-miss), dev/test split
             grouped by entity, metrics validated on a public benchmark, paired CIs
[ ] Phase 2  thin baseline, artifacts per stage, deterministic IDs, parse + chunk audits,
             traces, first LOG row per segment
[ ] Phase 3  loop: ladder → cheapest lever for the lowest failing rung → one change →
             paired dev test → confirm one level down → adopt / opt-in / reject + HLD
[ ] Phase 4  latency budget per stage, caches, time budgets + degradation flags,
             index type by scale + index recall, CI regression gate
```

---

## Where this project stops, and what would come next

| Item | Status | Why it matters |
| --- | --- | --- |
| 2.2 judge calibration | Built; 4/50 labels | Judge scores stay provisional until about 30+ human labels agree with them |
| 2.1 stronger generator (Claude) | Needs a key and a small paid run | The one experiment that tests the rung-5 bottleneck directly |
| 4.1 skip the planner for easy questions | Todo | The next latency win after the time budget |
| 3.7b LLM-written chunk context | Not run (~59M tokens, ~47 h local) | Needs an approved paid run |
| 3.1 step 2: table headers per row | Deferred | Worth it only once the generator can use it |

## Map of the documents

- **Walkthroughs**, step by step with code: [phase 1](walkthroughs/phase-1-baseline.md),
  [phase 2](walkthroughs/phase-2-generation.md),
  [phase 3](walkthroughs/phase-3-improvements.md),
  [phase 3 part 2](walkthroughs/phase-3-part-2-planning-and-reranking.md),
  [phase 4](walkthroughs/phase-4-latency-and-ingestion-experiments.md).
- **HLDs 001-028**, one per change, in `docs/hld/`.
- **ADRs**, for hard-to-reverse choices: dataset, embedder and store, rerankers, LLM
  clients.
- **[`experiments/LOG.md`](../experiments/LOG.md)**: every run, with its SHA, config and
  numbers.
- **[`backlog.md`](backlog.md)**: the status of every item.
