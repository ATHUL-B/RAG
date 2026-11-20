# CLAUDE.md

Guidance for Claude (and any AI/human contributor) working in this repository.
Read this file fully before making changes. When a rule here conflicts with a
quick shortcut, follow the rule.

## 0. What this project is

A **learning project**: build a production-grade Retrieval-Augmented Generation
(RAG) system step by step, and understand *why* each piece exists. The goals are:

1. Build every stage (parse → chunk → embed/index → rewrite → filter → hybrid
   retrieve → rerank → assemble → generate) as a swappable component.
2. Measure everything: retrieval metrics, generation metrics, latency and cost,
   on a fixed golden set, so every change is judged by numbers.
3. Be able to **localize any failure** to ingestion, retrieval, reranking,
   context assembly or generation (the "debugging ladder").
4. Learn how to scale it: latency budgets, caching, index tuning, freshness.

The study guide `RAG from Basics to Advanced A Production-Scale Study Guide.md`
is the curriculum. Its Part numbers are referenced throughout (e.g. "Part 9" =
hybrid search + RRF). The learning-path checklist in Part 15 is the roadmap.

## 1. Working Principles

- **Teach while building.** The user is learning. For every non-trivial change,
  explain the concept, link the guide Part, and say what metric should move and
  why. Prefer clear, readable implementations of a technique over a black-box
  framework call, at least the first time the technique appears.
- **Understand before changing.** Read the relevant code, `docs/` and guide Part.
- **Smallest correct change.** Do what was asked; no drive-by refactors or
  speculative features. Note unrelated issues in `docs/backlog.md` instead.
- **Ask when ambiguous.** If requirements are unclear or a decision is
  irreversible (vector DB, embedding model, dataset), ask rather than guess.
- **Be honest in reports.** State what was run, what passed, what failed, and
  what was skipped. Never claim a metric improved without showing the numbers.

## 2. RAG-Specific Rules (the core of this repo)

### Evaluation first
- **No retrieval/generation change without an eval run.** Every change to
  chunking, embeddings, filters, fusion, reranking or prompts is accompanied by
  a golden-set run comparing before vs. after.
- **Tune on dev, confirm on test.** Evaluation scripts default to the dev split
  (`evals/financebench-split.json`, grouped by company). Run `--split test` rarely, to
  confirm a change generalises; never choose settings by their test score, and never
  regenerate the split after seeing test results.
- **One change at a time** (ablation ladder, Part 13). Record each run in
  `experiments/LOG.md`: date, git SHA, config diff, recall@K, nDCG@K, MRR,
  faithfulness, refusal accuracy, p50/p95 latency, cost per query.
- **Report per segment**, not just the average. Golden questions are tagged by
  type (single-fact, multi-evidence, table/numeric, version-sensitive,
  exact-identifier, unanswerable, follow-up).
- **Retrieval and generation are evaluated separately.** Retrieval metrics
  never depend on the LLM's answer.

### Debuggability: every stage leaves evidence
- Each ingestion stage writes an **inspectable artifact** (JSONL/Parquet) under
  `data/processed/<stage>/`: parsed documents, cleaned text, chunks, embeddings
  metadata. You must be able to open a chunk and read exactly what was indexed.
- **Stable, deterministic IDs.** `doc_id` from the source; `chunk_id` derived
  from `doc_id` + char offsets + chunker config hash. Re-running ingestion with
  the same config yields the same IDs.
- **Label evidence at the source level** (doc + page + char span), not by
  chunk ID, and map spans to chunks at eval time. Golden labels must survive
  re-chunking.
- Every query produces a **trace**: rewritten queries, filters applied,
  dense/sparse candidates with scores and ranks, fused ranks, rerank scores,
  final context, prompt token count, answer, citations, and per-stage timings.
  Traces go to `runs/<run_id>/traces.jsonl`.
- The eval runner reports **which ladder rung failed** for each wrong answer:
  1. evidence not in the index → ingestion (parse/chunk)
  2. not in first-stage candidates → query/filter/embedding/hybrid
  3. dropped by reranker → rerank depth/model
  4. not in final prompt → context assembly
  5. in prompt but answer wrong → generation/prompt

### Records and reproducibility
- Every stored vector records the **embedding model ID and version**, chunker
  config hash and ingestion timestamp. Never mix models in one index.
- All pipeline settings live in versioned config (`configs/*.yaml`), never as
  literals in code. A run is reproducible from `git SHA + config + dataset version`.
- Measure **index recall** (ANN vs. exact search) and **candidate-pool recall**
  (before rerank) as their own metrics.

### Generation
- Answers must cite chunk/source IDs; uncited claims are a bug.
- There is always a **refusal path** when evidence is insufficient.
- LLM-as-judge: use a rubric, guard against position/verbosity bias, prefer a
  different model family from the generator, and calibrate against a small
  hand-labelled set (Part 13).

## 3. Repository Layout (target)

```
src/rag/
  ingestion/      # loaders, parsers, cleaners, chunkers, metadata enrichers
  indexing/       # embedders, vector store adapter, keyword (BM25) index
  retrieval/      # query rewrite, filters, dense, sparse, fusion (RRF), rerankers
  generation/     # context assembly, prompts, LLM client, citations, refusal
  evaluation/     # golden-set schema, retrieval & generation metrics, judges,
                  # runner, failure localization, reports
  observability/  # tracing, timers, token/cost accounting
  config/         # typed config models + loading/validation
  pipeline.py     # composes stages from config
configs/          # YAML configs per experiment (baseline.yaml, hybrid.yaml, ...)
evals/            # golden sets (small, versioned, committed) + labelling notes
experiments/      # LOG.md ablation history, analysis notebooks
scripts/          # CLI entry points: download, ingest, eval, serve
tests/            # mirrors src/
data/             # GITIGNORED: raw/, processed/, indexes/
runs/             # GITIGNORED: per-run traces, metrics, reports
docs/             # architecture, backlog, HLDs, ADRs, learning notes
```

Each component sits behind an abstract interface (`Parser`, `Chunker`,
`Embedder`, `VectorStore`, `KeywordIndex`, `Retriever`, `Fuser`, `Reranker`,
`Generator`, `Judge`) so implementations can be swapped by config for ablations.

## 4. Dataset

- Primary corpus and golden set are chosen in `docs/adr/0001-dataset.md`.
  Default recommendation: **FinanceBench** (real SEC filing PDFs with tables,
  multiple years per company, labelled evidence pages) as the main end-to-end
  corpus, plus **BEIR** subsets (SciFact, FiQA) to sanity-check retrieval
  metrics against published baselines. See README for the full list.
- Raw data is downloaded by `scripts/download_*.py`, never committed. Record
  source URL, version/commit and licence in `data/README.md`.
- Our own hard questions (version traps, unanswerable, multi-evidence) are added
  to `evals/` with tags; every production-style failure found becomes a new case.

## 5. Code Structure & OOP

- Python 3.12, managed with `uv`. Type-annotate public APIs.
- Prefer classes and interfaces where they model the domain (pipeline stages);
  use dataclasses/pydantic models for plain data (`Document`, `Chunk`,
  `Candidate`, `RetrievalResult`, `Trace`, `GoldenItem`).
- Favour composition over inheritance; inject collaborators (embedder, store,
  LLM client) so units are testable with fakes.
- Keep side effects (I/O, network, model calls, time, randomness) at the edges.
  Core logic (chunking, RRF, metrics) is pure and deterministic.
- No magic numbers: K, overlap, RRF k, rerank depth, thresholds come from config.
- Docstrings explain *why* (and cite the guide Part), not *what*.

## 6. Testing

- Every feature ships with tests: happy path, edge cases, error handling. Every
  bug fix gets a regression test.
- **Metrics are tested against hand-computed examples** (e.g. the nDCG@4 = 0.68
  and RRF examples from the guide). A wrong metric silently invalidates every
  experiment.
- Unit tests never call real models or networks: use fake embedders (e.g.
  hashed bag-of-words), in-memory stores and stub LLMs. Integration tests that
  need real models are marked `@pytest.mark.slow` and excluded by default.
- Mirror source layout under `tests/`. Do not skip, delete or weaken tests to
  make them pass; fix the cause.

## 7. Git Workflow

- `main` is always releasable: no direct pushes. One branch per feature/fix:
  `<type>/<short-description>` (e.g. `feat/bm25-index`, `exp/chunk-size-sweep`).
- Commit small and often; Conventional Commits (`feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `chore:`, `perf:`, `exp:` for experiments).
  Subject imperative, <= 72 chars; body explains *why*.
- One PR per feature. Experiment PRs include the before/after metrics table.
- Never commit secrets, datasets, indexes, model weights or run outputs.

## 8. Documentation (`docs/`)

Updated **in the same PR** as the change it describes.

```
docs/
  architecture.md   # current pipeline, components, data flow, key decisions
  backlog.md        # prioritized items with status (todo / in-progress / done)
  hld/              # design per feature before implementing it
  adr/              # hard-to-reverse choices: dataset, vector DB, models
  learning-notes/   # concept notes in the user's words, per guide Part
```

Keep `README.md` accurate: purpose, setup, how to ingest, how to run evals.

## 9. Observability, Latency & Cost

- Structured, leveled logging (no `print` in library code); never log secrets.
- Time every stage; report p50/p95/p99 per stage and end-to-end, plus
  time-to-first-token. Track tokens in/out and cost per query.
- Latency work follows Part 14: measure first, then parallelize retrievers,
  cap rerank depth, cache (embedding, retrieval, answer caches keyed with
  permissions/filters), skip rewrite for easy queries.
- External calls have timeouts, retries with backoff, and graceful fallbacks
  (e.g. reranker down → return fused ranking, flagged in the trace).

## 10. Configuration & Security

- Config from YAML + environment variables; validate at startup. Commit only
  `.env.example`; API keys never in code, configs or logs.
- Metadata filters for access control are **hard pre-filters**, never
  post-hoc text instructions to the LLM (Part 8).
- Treat retrieved document text as untrusted (prompt-injection aware prompts).

## 11. Working with AI Agents (Claude)

- Plan before coding for non-trivial tasks; share the plan when scope is large.
- Verify, don't assume: run tests, linters and the eval before declaring done.
- No destructive or outward-facing actions (force-push, deleting data/indexes,
  publishing, paid bulk API jobs) without explicit approval. Estimate cost
  before any large embedding/LLM batch run.
- Update this file when a new convention is agreed.

## 12. Definition of Done

1. Implemented behind the right interface, configurable, with traces/timings.
2. Tests cover it; full suite, lint (`ruff`), format and type checks (`mypy`) pass.
3. If it affects retrieval or generation: eval run recorded in `experiments/LOG.md`
   with per-segment metrics, latency and cost, compared against the previous run.
4. `docs/` (architecture, backlog, HLD/ADR, learning note) reflect the change.
5. Small conventional commits on a feature branch; focused PR with a clear
   description.
