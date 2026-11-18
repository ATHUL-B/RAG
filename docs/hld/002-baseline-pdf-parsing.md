# HLD 002: Baseline PDF parsing, artifacts and parse audit

## Goal

Backlog 1.3. Turn the FinanceBench PDFs into page-preserving `Document`s, write each
to an inspectable artifact, attach filter metadata, and **measure** how faithfully the
parser extracted the evidence (study guide, Part 3; debugging-ladder rung 1).

## Non-goals

- Cleaning (headers/footers, hyphenation) and table-aware parsing: backlog 3.1. The
  baseline is deliberately naive so later improvements show up in the numbers.
- Chunking (1.4).

## Verified facts (2026-11-18)

- FinanceBench `evidence_page_num` == pypdf zero-based page index (50/50 probed
  evidence pages matched at offset 0 with ≥ 0.8 token coverage).
- Some filings are AES-encrypted with an empty user password → `pypdf[crypto]`.

## Design

| Component | File | Role |
| --- | --- | --- |
| `PypdfParser(Parser)` | `ingestion/pdf_parser.py` | One `Page` per PDF page, raw `extract_text()`. A page that fails to extract becomes empty text (indices stay aligned) and is counted in `metadata["failed_pages"]`. Unreadable file → `IngestionError`. |
| `StaticMetadataEnricher(MetadataEnricher)` | `ingestion/enrichers.py` | Merges metadata from a `doc_id → Metadata` mapping (FinanceBench catalog: company, sector, doc_type, year). Unknown doc → `IngestionError` (filters must never silently miss a field). |
| `JsonArtifactStore[M]` | `storage.py` | One JSON file per key, atomic writes, key validation (no path traversal). Reused later for chunks. |
| `ParseStage` | `ingestion/parse_stage.py` | Orchestrates parse → enrich → save per document; idempotent (skips existing unless `force`), one failure doesn't stop the run, returns a `ParseReport` with per-document timings. |
| `build_parser(config)` | `ingestion/factory.py` | Config → implementation (ablations swap here). |
| `ParseAuditor` | `evaluation/parse_audit.py` | For each golden evidence span, token coverage of the annotator's `evidence_text` in the parsed page; summary + worst cases. |

Artifacts: `data/processed/parsed/<doc_id>.json`.

Config:

```yaml
parsing:
  parser: pypdf
evaluation:
  parse_audit_min_coverage: 0.8
```

### Parse audit metric

`coverage = |evidence tokens present in page tokens| / |evidence tokens|`, tokens =
lowercase alphanumeric runs. It is order-insensitive on purpose: it asks "did the text
survive extraction?", not "is it in reading order" (that matters for chunking and is
visible by reading the artifact). An evidence item **passes** if coverage ≥ threshold.
Evidence whose document was not parsed is reported separately, not as a 0.
Known limitation: number formatting differs between annotator text and extraction
(`5363` vs `5,363` tokenize differently), so coverage slightly understates fidelity on
numeric evidence; read the worst cases before concluding the parser lost text.

## Alternatives

- PyMuPDF: faster and better layout, but AGPL. pdfplumber / unstructured / docling:
  better tables, heavier. Candidates for backlog 3.1, compared on this same audit.

## Test plan

Unit tests build tiny real PDFs in memory (no fixtures on disk, no mocks of pypdf):
multi-page text, page order, encrypted-file error path, corrupt file. Enricher, store,
stage (with fake parser), auditor (hand-computed coverage) tested with fakes.
