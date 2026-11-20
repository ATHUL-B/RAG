# Phase 3, part 2: query planning and LLM reranking

Part 1 (`phase-3-improvements.md`) ended at dev recall@5 **0.427**, with a list of open
questions:

- Are small gains real?
- Why do rerankers hurt?
- How do we handle forward-looking questions?

Part 2 answers them. It follows the same loop: **find the biggest failure in the traces,
change one thing, measure, keep or reject.** Two things are new. First, every decision is
now made on a **dev split** and confirmed once on a held-out **test split**. Second, a
**local LLM** (Ollama) joins the pipeline twice: once to plan searches, once to rerank.

Read part 1 first; this document uses its vocabulary (traces, the debugging ladder,
ablation configs, RRF, pool recall). Study-guide references look like **(Part 7)**.

---

## 1. TL;DR

| # | Step | Backlog | Question it answered | dev recall@5 (n=96) | test (n=54) | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | **Dev/test split** (no pipeline change) | discovered | Are we fooling ourselves by tuning on what we report? | 0.427 | 0.426 | yes (method) |
| 6a | Read "FY22" as a year | 3.6 | Are short fiscal years lost? | 0.427 | | part of 6 |
| 6b | Forecast questions also search the prior year's filing | 3.6 | Can we fix "expected in FY2023"? | 0.438 | | part of 6 |
| 6c | **Strip instruction boilerplate** (+ original, RRF) | 3.6 | Is the instruction text drowning the question? | **0.469** ✱ | | part of 6 |
| 6d | Expand abbreviations (PP&E, COGS, …) | 3.6 | Does the embedder miss finance shorthand? | 0.432 | | part of 6 |
| 6 | **Rule-based planner** (6a-d together) | 3.6 | | **0.484** ✱ | **0.481** ✱ | **yes: default** |
| 7 | **Local LLM planner**: sub-queries per fact | 3.6 | Can one search per needed fact fix multi-evidence questions? | **0.583** ✱ | 0.546 (n.s.) | opt-in |
| 8 | Weight the original question in RRF (1.5 / 2 / 3) | 3.6 | Are the planner's losses caused by outvoting? | 0.594 / 0.557 / 0.573 | not run | no (rejected) |
| 9a | bge cross-encoder over the planner's pool | 3.5 | Did rerankers fail because the pool was poor? | 0.510 | | no (worse) |
| 9 | **Local LLM reranker** (pointwise grades, top 20) | 3.5 | Can an LLM rank the evidence into the top 5? | **0.740** ✱ | **0.667** ✱ | **yes: best quality** |

✱ = significant against the previous kept step (paired bootstrap, 95% CI excludes 0).

**On test, recall@5 went from 0.426 to 0.667 (+0.241, CI [+0.111, +0.361]).** The right
page is now in the top 5 for about two-thirds of questions instead of about two-fifths.

Two configs, two purposes:

| Config | dev / test recall@5 | Latency per question | Needs |
| --- | --- | --- | --- |
| `planner-rules.yaml` (**default**) | 0.484 / 0.481 | ~16 ms | nothing |
| `planner-llm-rerank-llm.yaml` (**best quality**) | 0.740 / 0.667 | ~27 s uncached | Ollama + `llama3.1:8b` |

---

## 2. What changed in the method

### 2.1 Dev/test split first, before any new knob

Part 1's lesson 4 was "don't tune on the test set". So far that only meant "leave knobs at
their defaults". Query rewriting brings many real knobs (prompt wording, number of
sub-queries, fusion weights). Tuning them on the 150 questions we report on would overfit
the evaluation, the same way training on the test set overfits a model (Part 13).

**`evaluation/splits.py`: `make_split()`**

```python
groups = {company: [items]}  # all of a company's questions stay together
for company in sorted(groups, key=seeded_hash):  # deterministic "random" order
    if adding it brings test's strata closer to 1/3 of each stratum:
        put the company in test
    else:
        put it in dev
```

*Intention, line by line:*

- **Grouped by company.** Questions about one company share filings, wording and table
  layouts. A random split would let dev tuning "see" the test companies. Grouping makes
  test measure *companies we never tuned on*, which is what a new filing looks like in
  production.
- **Stratified** by question type × single/multi evidence, so both halves have the same
  mix of difficulty.
- **Seed 0, fixed before looking.** Picking the seed whose split "looks nicest" is itself
  a mild bias.
- **Committed as IDs** in `evals/financebench-split.json` (96 dev / 54 test, 13 test
  companies). `scripts/make_split.py` refuses to overwrite it without `--force`.

`scripts/evaluate_*.py --split dev|test|all` defaults to **dev**. CLAUDE.md gained the
rule: tune on dev, run test rarely, never choose by test.

**What it cost:** 96 questions resolve only differences of about 0.07 or more. That is why
§1 marks some steps "part of 6": individually they are ties.

### 2.2 How each step was chosen

Each result pointed at the next bottleneck, as in part 1:

```
pages-context: long questions are mostly instructions               → strip boilerplate (6)
rules:         multi-evidence questions still at 0.26                → one search per fact (7)
LLM planner:   4 losses; hypothesis "original outvoted"              → weight original (8)
weighting:     rejected; evidence was rank 3-5, pool_recall 0.976    → rerank the pool (9)
```

---

## 3. What changed in the code, at a glance

### 3.1 New and changed modules

| Module | New / changed | Step | Purpose |
| --- | --- | --- | --- |
| `evaluation/splits.py`, `scripts/make_split.py` | new | 5 | Company-grouped, stratified dev/test split |
| `app.py: load_golden_items` | new | 5 | One place that picks a split's questions |
| `retrieval/filters.py` | changed | 6a, 6b | `short_fiscal_years`, `forward_looking_prior_year` |
| `retrieval/rewriting.py` | new | 6c, 6d, 7 | `strip_boilerplate`, `expand_abbreviations`, `RuleBasedRewriter`, `CompositeRewriter` |
| `retrieval/multi_query.py` | new | 6, 8 | `MultiQueryRetriever`: search original + rewrites, fuse with RRF |
| `retrieval/llm_planner.py` | new | 7 | `LLMQueryPlanner`, `PlanCache` |
| `retrieval/llm_reranker.py` | new | 9 | `LLMPointwiseReranker`, `GradeCache` |
| `generation/llm_clients.py` | changed | 7, 9 | `check_ollama_model`, `num_predict`, `num_ctx` |
| `config/settings.py` | changed | all | `RewriteConfig`, `LLMPlannerConfig`, `RerankerKind`, `LLMRerankConfig` |
| `retrieval/base.py: RetrievalResult` | changed | 6 | `queries`: the search texts actually used |

### 3.2 How retrievers compose now

```
FilteredRetriever                    filters from the ORIGINAL question (rules)
  └─ RerankingRetriever              (optional) take 20, rerank, keep k
       │    reranker: CrossEncoderReranker | LLMPointwiseReranker
       └─ MultiQueryRetriever        (optional) original + rewrites → RRF
            │    rewriter: CompositeRewriter[RuleBasedRewriter, LLMQueryPlanner]
            └─ Dense | BM25 | Hybrid
```

The order is deliberate:

- **Filters outside the rewriter.** They are extracted once, from the original question,
  and every rewritten search uses the same ones. An LLM-guessed filter could silently
  hide the evidence (Part 7's main risk), so filters stay rule-based.
- **Reranker outside the rewriter.** It reorders the *fused* pool against the *original*
  question, not against a rewrite.
- **Composition root builds the LLM parts.** `app.py: build_planner` and
  `build_llm_reranker` construct the planner and reranker (they need the cache paths in
  `PathsConfig`). They are injected into `build_retriever`, which raises `ConfigError` if
  a config asks for one without it. The factory stays testable with fakes.

---

## 4. Step 6: the rule-based query planner (backlog 3.6, HLD 011)

### Problem the data showed

Sizing on dev only, before writing any rule:

| Pattern in the question | Dev questions | Not hit today |
| --- | --- | --- |
| Instruction boilerplate ("Assume you are…", "Round your answer…", "If X is not a useful metric… explain why") | 25 | 18 |
| Abbreviations (PP&E, COGS, DPO, ROA, capex) | 14 | 12 |
| "FY22", "between FY20 and FY21" | 8 | 6 |
| Forecasts ("expected in FY2023": published in the FY2022 filing) | 5 | 3 |

### Code and intention

**`retrieval/filters.py`: two new switches on `CatalogFilterExtractor`**

```python
_SHORT_FISCAL_YEAR = re.compile(r"\bFY\s?'?(\d{2})(?![\d,.]\d|\d)", re.IGNORECASE)
_FORWARD_LOOKING = re.compile(r"\b(?:expect…|forecast…|guidance|outlook|anticipat…|project…)\b")
```

- `short_fiscal_years`: "FY22" → 2022. The `FY` prefix is required, because a bare "22" is
  usually a count or a percentage. The lookahead stops "FY2023" from also reading as FY20.
- `forward_looking_prior_year`: only when a forecast cue is present, also allow year − 1.
  *Intention:* part 1 rejected a blanket ±1 window, because it readmitted neighbouring
  years for every question. This widens only the questions that need it.

**`retrieval/rewriting.py: strip_boilerplate()`**

```python
text = _PARENTHETICAL_INSTRUCTION.sub("", question)  # "(…round to one decimal…)"
for sentence in _SENTENCE.split(text):  # sentences, and ';' clauses
    if not _INSTRUCTION.search(sentence):
        kept.append(sentence)
    elif tail := _QUESTION_TAIL.search(sentence):  # "Based on X, what is …?"
        kept.append(tail.group(1))  # keep the question part
if not kept:
    kept = [question]  # never empty a query
titles = [title for pattern, title in _STATEMENTS if pattern.search(question)]
```

- **Not everything that looks like boilerplate is noise.** Statement names tell us *which
  page* to look for, so they are kept, rewritten to the titles filings use ("P&L
  statement" → "consolidated statements of income operations"). Definitions ("DPO is
  defined as: 365 × average accounts payable…") name the line items, so they are kept too.
- **Two bugs were caught by reading the output on dev before any run.** A question lost
  entirely because "round to" sat inside brackets in the question sentence; fixed by
  removing bracketed asides first. And "…for FY2022; if not meaningful, explain why" lost
  a useful clause; fixed by splitting on `;`.

**`expand_abbreviations()`** appends the spelled-out name of each abbreviation found
(16-term glossary). Names only, never facts (Part 7, rule 2).

**`retrieval/multi_query.py: MultiQueryRetriever`**

```python
rewrites = rewriter.rewrite(query)  # timed as the "rewrite" stage
if not rewrites:
    return inner.retrieve(query, k)  # no change → no extra cost
searches = {"original": query, "rewrite_1": …, …}  # keep_original (Part 7 rule 1)
results = {name: inner.retrieve(q, depth) for …}
fused = fuser.fuse([label(r, name) for …], k)  # RRF; lists labelled by name (step 8)
```

*Intention:* a rewrite can drift, so the original question stays one of the searches and
anchors the fusion.

### Result and lesson

| dev (n=96) | recall@5 | Δ vs 0.427 | better/worse |
| --- | --- | --- | --- |
| 6a FY22 | 0.427 | 0 | 0/0 |
| 6b forecasts | 0.438 | +0.010 | 1/0 |
| **6c boilerplate** | **0.469** | **+0.042 ✱** | **6/0** |
| 6d abbreviations | 0.432 | +0.005 | 1/0 |
| **6 all four** | **0.484** | **+0.057 ✱** | **8/0** |
| **6 on test** | **0.481** (from 0.426) | **+0.056 ✱** | **4/0** |

- **The instructions were the problem.** Long calculation questions are mostly
  instructions, and removing them helped most where we were weakest: multi-evidence
  0.143 → 0.262, metrics-generated 0.550 → 0.700.
- **A correct fix that doesn't move recall is still correct.** AMD "FY22" questions now
  filter to 2022, but AMD has only one filing, so the company filter had already selected
  it. Traces showed the remaining AMD misses are page ranking *within* that filing.
- **Became the default.** It is significant on both splits, never made a question worse,
  and costs about 8 ms.

---

## 5. Step 7: the local LLM query planner (backlog 3.6, HLD 012)

### Problem

After step 6, multi-evidence questions were still at 0.262. A DPO ratio needs accounts
payable (balance sheet) *and* cost of sales (income statement). One question vector sits
between them, close to neither.

### Code and intention

**`retrieval/llm_planner.py: LLMQueryPlanner`**, a `QueryRewriter`:

```python
PLANNER_SYSTEM = """… One query per distinct fact, line item or financial statement …
Never include numbers, values or facts that are not in the question. Do not answer."""
schema = {"sub_queries": [string, maxLength=max_chars], 1..max_sub_queries}
```

- **Reuses `OllamaLLMClient`** from generation (`temperature: 0`, schema-constrained
  output through Ollama's `format`). No new HTTP code: the `LLMClient` interface from
  Phase 2 paid off.
- **`_validated()`** normalises whitespace, deduplicates case-insensitively, drops copies
  of the question (compared *before* truncation, a bug the unit tests caught: a cut copy
  "What is Wa…" slipped through), and caps count and length.
- **Failure means "no sub-queries".** A timeout, an API error or bad JSON means retrieval
  falls back to the rules. The failure is counted and never cached.
- **`PlanCache`**: `data/cache/planner/<model>-<hash(prompt + schema)>.json`. Re-runs are
  free and identical, and changing the prompt or schema automatically starts a new cache.
- **`check_ollama_model()` preflight** in `evaluate_retrieval.py`. *Intention:* without
  it, a run where Ollama is down would fail every plan, fall back to rules, and **look
  like a real "no improvement" result**. A silent fallback must never be mistaken for a
  measurement.

Example plan, from the smoke test:

```
Q: What is Amazon's FY2017 DPO? DPO is defined as: 365 * (average AP between FY2016 and FY2017) / …
→ "Amazon FY2017 accounts payable balance sheet"
  "Amazon FY2016 accounts payable balance sheet"
  "Amazon FY2017 cost of goods sold income statement"
  "Amazon FY2016 change in inventory balance sheet"
```

### A bug only a real run could find

One question hit the 120 s timeout **every time**. Calling Ollama directly showed why:
the model wrote three good sub-queries, then looped inside a string ("…to get correct
revenue figure, if necessary, and to get correct revenue figure…") and never closed the
JSON. Two fixes:

- `maxLength` per sub-query in the **schema**. Ollama compiles the schema into a grammar,
  so the model is *forced* to close the string, and the good sub-queries survive. The
  question went from 120 s to 3.4 s.
- `num_predict` (an output-token cap) as a backstop for any other runaway.

The schema changed, so the cache key had to change too. Old plans were made under the
old schema.

### Result and lesson

| | `planner-rules` | `planner-llm` | Δ | better/worse |
| --- | --- | --- | --- | --- |
| dev | 0.484 | **0.583** | **+0.099 ✱** | 16/4 |
| test | 0.481 | 0.546 | +0.065 (n.s.) | 11/5 |

- Multi-evidence 0.262 → 0.524 (dev): decomposition did what it was designed for.
- **An audit of all sub-queries found no invented values.** The model does add prior years
  for trend questions ("improving as of FY2022" → FY2020 and FY2021 searches). That is a
  reasonable inference, and it only changes search text, never filters.
- **Not significant on held-out companies, ~2.3 s per question, Ollama required** →
  opt-in, not default.

---

## 6. Step 8: weighting the original question (HLD 013), a negative result

### Hypothesis

In the planner's 4 losses, sub-queries that agree with each other outvote the original
question's correct page in RRF. Give the `original` list a higher weight.

### Code and intention

`MultiQueryRetriever` now labels each list's candidates with its search name before
fusing, so `ReciprocalRankFusion(weights={"original": w})` can target it.
`rewrite.original_weight` defaults to 1.0, and a re-run reproduced 0.583 *to the digit*:
the labelling alone changes nothing.

### Result and lesson

| weight | 1.5 | 2 | 3 |
| --- | --- | --- | --- |
| dev recall@5 | 0.594 (tie) | 0.557 | 0.573 |
| losses recovered | 0/4 | 0/4 | 1/4 |

**Rejected. The interesting part is why.** `stage_ranks` in the traces (part 1's
evaluation addition) showed the original question had the evidence at **rank 3-5, never
rank 1**. It was barely inside the top 5 even under the rules. More lists bring more
strong pages that push marginal pages out. Weighting the original lifts its rank-1 and
rank-2 pages, which are mostly *not* the evidence.

Meanwhile `pool_recall` was **0.976**: the evidence was in the fused candidates for almost
every question. **Finding was solved; ordering was the bottleneck.** Test was not run,
because no change was chosen.

---

## 7. Step 9: the local LLM reranker (backlog 3.5, HLD 014)

### Measured before building

A reranker can only reorder what it is given. Fused recall of `planner-llm` at depth k
on dev:

| k | 5 | 10 | 15 | **20** | 30 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| recall | 0.583 | 0.745 | 0.807 | **0.859** | 0.901 | 0.915 |

Depth 20: the ceiling is 0.859, and deeper adds little at linear cost.

### Code and intention

**`retrieval/llm_reranker.py: LLMPointwiseReranker`**

```python
for candidate in candidates:  # top 20 of the fused pool
    grade = self._grade(question, candidate)  # one Ollama call → {"grade": 0..3}
    if grade is None:
        return candidates[:top_n]  # any failure → fused order
order = sorted(range(n), key=lambda i: (-grades[i], i))  # ties keep fused order
```

- **Pointwise, not listwise.** Listwise rerankers (read 20 passages, output an order)
  work for large models. Small models suffer from position bias and "lost in the middle".
  One passage per call avoids both.
- **The rubric:**
  - 3 = contains a needed figure or fact; for multi-fact questions, any one counts.
  - 2 = right statement but not the figure.
  - 1 = same company, other topic or period.
  - 0 = irrelevant.
- **Ties keep the fused order**, so the reranker can only *promote* passages; it never
  scrambles the rest.
- **The passage is data, not instructions** (prompt-injection aware, CLAUDE.md §10).
  `num_ctx` is set explicitly, because Ollama otherwise **silently truncates** long
  prompts from the front.
- **`GradeCache`** per (question, chunk), keyed by model + prompt + schema + layout
  version.

### The decision that mattered most: prompt layout, measured on dev

The first smoke test was alarming: 18 of 20 passages graded 0, *including both evidence
tables*. Before spending an hour on a full run, three layouts were compared on 12 dev
questions, using P(evidence graded above non-evidence) as the measure:

| Layout | Pairwise accuracy |
| --- | --- |
| question first, then passage | **0.65**: *worse than no reranker* |
| fused order (no reranker) | 0.67 |
| passage, one-sentence reason, then grade | ~0.75, but 4× slower |
| **passage first, question last** | **0.75** (chosen) |

A small model attends most to what it read last. Without this check, the full run would
have made rankings *worse*. `grader_user_message()` documents the measurement, and
`GRADER_LAYOUT` is part of the cache key.

### Result and lesson

| | `planner-llm` | + bge cross-encoder | **+ LLM reranker** |
| --- | --- | --- | --- |
| dev | 0.583 | 0.510 (nDCG/MRR ✱ worse) | **0.740** (+0.156 ✱, 21/3) |
| test | 0.546 | | **0.667** (+0.120 ✱, 8/1) |
| rerank p50 | | 0.64 s | ~26 s |

- **Part 1's open question "why do rerankers hurt?" is answered.** The bge cross-encoder is
  still worse on a pool that contains the evidence 97.6% of the time, so it was the model,
  not the pool. Web-trained cross-encoders don't read flattened financial tables. An LLM
  that reads the question and the table together does.
- **Every segment improved** on dev. Logical-reasoning questions went from 0.389 to 0.556,
  and they had been at 0.000 before the planner.
- **The 3 losses** are evidence tables the 8B grader scored 0. One (Best Buy) may be a
  page-label gap: the page graded 3 likely also contains the store count.
- **Cost:** 20 calls × ~1.3 s ≈ 26 s per question. Fine offline, too slow interactively.
  Hence "best quality", not default.

---

## 8. How the evaluation evolved in part 2

| Added | Where | Why it was needed |
| --- | --- | --- |
| Dev/test split, `--split`, split recorded in summaries | `evaluation/splits.py`, scripts | Tune on dev, confirm once on test (step 5) |
| `queries` in every trace | `RetrievalResult`, `QueryTrace` | See the exact search texts a rewrite produced (step 6) |
| `rewrite` stage timing | `MultiQueryRetriever` | The LLM planner makes rewriting the slow part (step 7) |
| Planner / reranker counts in `metrics.json` | `evaluate_retrieval.py` | Calls, cache hits, failures or fallbacks: a run with silent fallbacks is not a measurement |
| Ollama preflight | `check_ollama_model` | Fail fast instead of producing a misleading run |
| Depth curve before a reranker | ad-hoc analysis, HLD 014 | Pick the depth from the recall ceiling, not by habit |
| Pairwise accuracy on a small dev sample | ad-hoc analysis, HLD 014 | Check a prompt is better than "no change" before an hour-long run |

Lessons, adding to part 1's:

1. **Split before you tune.** The LLM planner looked strong on dev (+0.099) and was not
   significant on test. Without the split, we would have adopted it as a clear win.
2. **A negative result with a good diagnosis is worth as much as a win.** Step 8 failed,
   but its trace analysis (rank 3-5, pool recall 0.976) pointed straight at step 9.
3. **Every LLM component needs a fallback, a cache and a counter.** Fallbacks keep the
   pipeline working when the model fails; counters keep the *measurement* honest when it
   does; caches make re-runs free and reproducible.
4. **Small prompt details can flip the sign of a result.** Question-first vs
   question-last turned a harmful reranker into the biggest gain of Phase 3.
5. **Real runs find bugs fakes can't.** The JSON loop only appeared with the real model.
   The question-copy bug only appeared in a unit test. Both kinds of test were needed.

---

## 9. Configs and how to reproduce

Each ablation differs from its parent in one section (enforced by
`tests/config/test_loader.py`):

```
pages-context.yaml                         (part 1 default; dev 0.427)
├─ planner-fy.yaml                         + filters.short_fiscal_years          (6a)
├─ planner-forward.yaml                    + filters.forward_looking_prior_year  (6b)
├─ planner-clean.yaml                      + rewrite.strip_boilerplate           (6c)
│  └─ planner-rules.yaml  ← DEFAULT        + 6a, 6b, 6d                          (6)
│     └─ planner-llm.yaml                  + rewrite.llm_planner                 (7)
│        ├─ planner-llm-w1-5 / -w2 / -w3   rewrite.original_weight               (8)
│        ├─ planner-llm-rerank-bge.yaml    + rerank (bge, depth 20)              (9a)
│        └─ planner-llm-rerank-llm.yaml    + rerank (kind: llm, depth 20)        (9) ← BEST QUALITY
└─ planner-abbrev.yaml                     + rewrite.expand_abbreviations        (6d)
```

All of these reuse part 1's chunks and index. Only retrieval changes.

```bash
# free, no local model
uv run python scripts/evaluate_retrieval.py --config configs/planner-rules.yaml            # dev
uv run python scripts/evaluate_retrieval.py --config configs/planner-rules.yaml --split test

# local model: install Ollama, then `ollama pull llama3.1:8b`
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm.yaml              # ~4 min first run
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm-rerank-llm.yaml   # ~40 min first run
uv run python scripts/compare_runs.py runs/planner-llm-dev-<ts> runs/planner-llm-rerank-llm-dev-<ts>
```

First runs are slow because every plan and grade is a model call. Re-runs read
`data/cache/planner/` and `data/cache/rerank/` and finish in seconds with identical
results. The caches are safe to delete.

---

## 10. Reading order for the part 2 code

1. `evaluation/splits.py`, then `app.py: load_golden_items`: how questions are chosen.
2. `retrieval/filters.py`: the two new year rules (short, readable regexes).
3. `retrieval/rewriting.py`: `strip_boilerplate`, then `RuleBasedRewriter` and
   `CompositeRewriter`; `tests/retrieval/test_rewriting.py` shows real dev questions
   before and after.
4. `retrieval/multi_query.py`: the fusion of original + rewrites, and the list labelling.
5. `retrieval/llm_planner.py`, with `tests/retrieval/test_llm_planner.py` (fake LLM, every
   failure mode).
6. `retrieval/llm_reranker.py`: `grader_user_message` and its docstring carry the key
   measurement.
7. `retrieval/factory.py`, then `app.py: build_planner` / `build_llm_reranker`: how config
   becomes the stack in §3.2.
8. `generation/llm_clients.py: check_ollama_model` and the `num_predict` / `num_ctx`
   options.

Design records: HLD 011 (rules), 012 (LLM planner), 013 (weighting, rejected), 014 (LLM
reranker), each with results and a decision.

---

## 11. Part 1's open questions: where they stand

| Part 1 question | Answer now |
| --- | --- |
| Are small gains real? | Now tested properly: dev for choosing, test for confirming. Small steps are still ties at n=96; combined steps are significant on both splits. |
| Why do rerankers hurt here? | **Answered:** the cross-encoders can't read financial tables (still worse on a rich pool). An LLM grader can (step 9). |
| Forward-looking questions | Rule 6b handles explicit cues. The LLM planner adds prior-year searches for trend questions. |
| Incomplete page labels | Still open: one reranker "loss" is probably a correct page without a label. Needs answer-level judges (Phase 2.2). |
| Tables as text | Still open, and now the main remaining cause of reranker errors (3.1). |
| Why does context hurt `metrics-generated`? | Superseded: metrics-generated is now the best segment (dev 0.867, test 0.800). |

## 12. What's next

- **Phase 2:** measure *answers*: a real generation run (free with Ollama), correctness
  and faithfulness judges (2.2), and our own unanswerable questions (2.3). Decide whether
  generation uses the default or the best-quality retrieval.
- **Phase 4:** the reranker's 26 s per question is the first real latency budget problem
  (Part 14): fewer candidates, parallel grading, a smaller grader, or distilling the grades
  into a fast cross-encoder.
- **3.1 table-aware parsing**, the common cause of the remaining reranker errors.
