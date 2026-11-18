# ADR 0001: FinanceBench as the primary corpus

- Status: accepted
- Date: 2026-11-18

## Context

We need a corpus that exercises every lesson in the study guide: messy PDFs with
tables (Part 3), version-sensitive retrieval (Part 8's 2022-vs-2024 trap), exact
identifiers (Part 9), multi-evidence questions, and labelled evidence so retrieval
can be measured separately from generation (Part 13).

## Decision

Use **FinanceBench** (Patronus AI, https://github.com/patronus-ai/financebench) as the
primary end-to-end corpus.

Verified facts (2026-11-18):
- 150 open-source questions, 84 distinct filings referenced; 361 filings described
  in `financebench_document_information.jsonl`, 368 PDFs (~705 MB) in `pdfs/`.
- Each question carries 1-3 evidence items with `doc_name`, **zero-based**
  `evidence_page_num`, `evidence_text` and the full page text.
- Three question types (50 each): `metrics-generated`, `domain-relevant`, `novel-generated`.
- Documents carry `company`, `gics_sector`, `doc_type`, `doc_period` (year) → filter metadata.
- Same company appears across years (e.g. 3M 2015-2023) → natural version traps.

Supporting datasets (added when their milestone arrives): BEIR SciFact/FiQA for
validating retrieval code against published numbers; MultiHop-RAG for multi-hop and
null queries; LegalBench-RAG for span-level clause retrieval; RAGTruth for judge
calibration; MS MARCO / NQ for scale and latency.

## Consequences

- Only 150 labelled questions and **no unanswerable ones**: we will write our own
  hard cases (version traps, unanswerable, multi-part) into `evals/`.
- Evidence is labelled at page level, so retrieval metrics are computed as
  "a chunk overlapping the evidence page (and containing the evidence text)".
- Financial tables make naive text extraction visibly fail: good for learning.
- Licence: the upstream repo has no LICENSE file; its README asks for citation of
  Islam et al. (2023), arXiv:2311.11944. Use for personal learning only; do not
  redistribute the data. Re-check before any other use.
- Raw data lives in `data/raw/financebench/` (gitignored), fetched by
  `scripts/download_financebench.py`.
