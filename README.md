# RAG
To learn production level RAG system

Curriculum: `RAG from Basics to Advanced A Production-Scale Study Guide.md`.
Contributor/agent guidance: `CLAUDE.md`. Design docs: `docs/` (start with
`architecture.md` and `backlog.md`). **New here? Read
[`docs/walkthroughs/phase-1-baseline.md`](docs/walkthroughs/phase-1-baseline.md)** for what
was built, why, and how to read the code, then
[`docs/walkthroughs/phase-3-improvements.md`](docs/walkthroughs/phase-3-improvements.md) and
[part 2](docs/walkthroughs/phase-3-part-2-planning-and-reranking.md) for how retrieval was
improved one measured change at a time, and
[`docs/walkthroughs/phase-2-generation.md`](docs/walkthroughs/phase-2-generation.md) for how
answers are generated, checked, judged and refused, and
[`docs/walkthroughs/phase-4-latency-and-ingestion-experiments.md`](docs/walkthroughs/phase-4-latency-and-ingestion-experiments.md)
for the latency budget, rerank time limits, table parsing, chunk sizes and parent-child retrieval.

**The summary of the whole project is in [`docs/final-write-up.md`](docs/final-write-up.md):**
every measured step from recall@5 0.273 to 0.781, what failed and why, and a playbook for
building and optimising RAG on any corpus at any scale.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

### Quick start: from a fresh clone to an evaluation

Data, indexes and run outputs are never committed (CLAUDE.md §4), so a fresh
clone rebuilds them locally. Rebuilding is free and deterministic: the same
config gives the same chunk IDs and the same index.

```bash
make install    # dependencies
make data       # download 84 PDFs (~160 MB), parse, chunk, embed with bge-small on CPU
cp .env.example .env    # then put your ANTHROPIC_API_KEY in .env
uv run python scripts/evaluate_retrieval.py --config configs/planner-rules.yaml   # free; dev split by default (--split test|all)
uv run python scripts/evaluate_generation.py --config configs/generation-claude.yaml --dry-run  # cost estimate
uv run python scripts/evaluate_generation.py --config configs/generation-claude.yaml --limit 10 # ~$0.40
```

Evaluation scripts use the **dev** split by default (`evals/financebench-split.json`,
96 of the 150 questions). Tune on dev; run `--split test` only to confirm a change
generalises (see `evals/README.md`).

`make data` builds the current default config (`configs/planner-rules.yaml`,
which `generation-claude.yaml` shares). To build the data for another
experiment, run `make data CONFIG=configs/<name>.yaml`. Steps that are already
done are skipped, so it is safe to re-run.

### Step by step (what `make data` runs, plus the audits)

```bash
make install                 # uv sync: creates .venv with all dependencies
make check                   # format check, lint, type check, tests (what CI runs)
make download-financebench   # metadata + the 84 PDFs the questions reference
uv run python scripts/parse_documents.py   # PDFs -> data/processed/parsed/<doc_id>.json
uv run python scripts/audit_parsing.py     # how much golden evidence survived parsing
uv run python scripts/chunk_documents.py   # -> data/processed/chunks/<fingerprint>/<doc_id>.json
uv run python scripts/audit_chunking.py    # how much evidence fits in a single chunk
uv run python scripts/build_index.py       # embed chunks (bge-small, local) -> data/indexes/dense/
uv run python scripts/evaluate_retrieval.py  # recall@K, nDCG, MRR per segment + traces in runs/
uv run python scripts/evaluate_retrieval.py --config configs/filters.yaml  # ablation: metadata filters
uv run python scripts/evaluate_retrieval.py --config <config with llm cache: false> --limit 24  # cold latency run (HLD 019); keep the Mac awake: caffeinate -dims
uv run python scripts/evaluate_retrieval.py --config configs/hybrid.yaml   # ablation: dense + BM25 (RRF)
uv run python scripts/evaluate_retrieval.py --config configs/rerank-bge.yaml  # ablation: + cross-encoder
uv run python scripts/evaluate_retrieval.py --config configs/planner-clean.yaml  # ablation: + query cleanup (HLD 011)
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm.yaml  # + local LLM planner: needs Ollama + `ollama pull llama3.1:8b`
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm-rerank-llm.yaml  # + local LLM reranker (best quality, ~27 s/question)
uv run python scripts/evaluate_retrieval.py --config configs/planner-llm-rerank-llm-budget13.yaml  # balanced: 13 s rerank budget (~16 s/question, recall@5 0.688; HLD 020)
# opt-in parser that keeps table rows on one line (HLD 021): parse/chunk/index once with `make data CONFIG=configs/planner-rules-layout.yaml`
# opt-in parent-child retrieval (HLD 023): needs the 200-token child index first: `make data CONFIG=configs/chunks-200-o0.yaml`, then configs/planner-llm-rerank-llm-pc.yaml
# default (no local model needed): configs/planner-rules.yaml; best quality: configs/planner-llm-rerank-llm.yaml (same chunks/index as pages-context.yaml)

# Phase 2: answers with citations (needs ANTHROPIC_API_KEY in .env, see .env.example)
uv run python scripts/evaluate_generation.py --config configs/generation-claude.yaml --dry-run
uv run python scripts/evaluate_generation.py --config configs/generation-claude.yaml --limit 10
# local model instead: install Ollama (https://ollama.com), `ollama pull llama3.1:8b`, then
uv run python scripts/evaluate_generation.py --config configs/generation-ollama.yaml --limit 10
uv run python scripts/evaluate_generation.py --config configs/generation-ollama.yaml --golden evals/unanswerable-v1.jsonl --limit 0  # 2.3: refusal recall
uv run python scripts/evaluate_generation.py --config configs/generation-ollama.yaml --golden evals/unanswerable-v2.jsonl --limit 0  # near-miss traps (HLD 026)
# stronger model via an OpenAI-compatible LLM gateway: follow docs/runbooks/gateway-generation-run.md
uv run python scripts/judge_generations.py runs/<generation-run>   # 2.2: correctness + faithfulness judges (needs `ollama pull qwen2.5:7b`)
uv run python scripts/judge_calibration.py label    # blind hand labels; then `report` for Cohen's kappa
uv run python scripts/compare_runs.py runs/<a> runs/<b>   # per-question diff + 95% bootstrap CI
uv run python scripts/budget_recall.py runs/<a> runs/<b> --baseline runs/<a>  # recall at an equal prompt-token budget (HLD 022)
make gate   # retrieval regression gate (CI job eval-gate): 32 dev questions, ~2 min; if a change moves them on purpose: uv run python scripts/eval_gate.py --update, and commit evals/gate-v1.json (HLD 024)
uv run python scripts/download_beir.py && uv run python scripts/evaluate_beir.py   # cross-check our retrieval + metrics against published MTEB scores (~7 min first run; HLD 025)
uv sync --group ann && uv run python scripts/index_recall.py   # HNSW index recall vs exact search, ours vs faiss, filtered search (~10 min; HLD 028)
```

Other commands: `make format`, `make test`, `make test-slow` (network/model tests),
`uv run python scripts/download_financebench.py --scope {metadata,questions,all}`.

## Layout

```
src/rag/      domain, ingestion, indexing, retrieval, generation, evaluation, observability, config, datasets
configs/      one YAML per experiment (baseline.yaml)
evals/        our own golden questions
experiments/  LOG.md: ablation history
scripts/      CLI entry points
tests/        mirrors src/
data/, runs/  gitignored
```

## Datasets

| Dataset | Role in this project | Why |
| --- | --- | --- |
| **FinanceBench** (Patronus AI) | Primary end-to-end corpus | Real SEC filing PDFs with tables; same company across years (version traps like the guide's 2022 vs 2024 policy); questions with labelled evidence pages |
| **BEIR** (SciFact, FiQA, NFCorpus) | Retrieval sanity check | Ready-made qrels; compare our nDCG@10 to published baselines to validate metric code, hybrid search and rerankers |
| **MultiHop-RAG** | Multi-evidence and refusal | News corpus with multi-hop queries and "null" (unanswerable) queries |
| **LegalBench-RAG** | Clause-level precision | Contracts/policies with character-span ground truth; closest to insurance-clause retrieval |
| **RAGTruth** | Faithfulness judge calibration | Human-labelled hallucinations in RAG answers |
| **MS MARCO / BEIR-NQ** | Scale and latency | Millions of passages for ANN tuning, index recall, caching, p95 latency |

Data is downloaded by scripts, never committed. Check each dataset's licence before use.
