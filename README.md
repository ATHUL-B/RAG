# RAG
To learn production level RAG system

Curriculum: `RAG from Basics to Advanced A Production-Scale Study Guide.md`.
Contributor/agent guidance: `CLAUDE.md`. Design docs: `docs/` (start with
`architecture.md` and `backlog.md`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

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
