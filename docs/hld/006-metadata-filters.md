# HLD 006: Metadata filters from the question

## Goal

Backlog 3.3, the first Phase 3 ablation. The baseline spends **40.7% of top-5 slots on
other filings of the same company** (the version trap, Part 8). Extract `company` and
`year` from the question, search only that slice of the index, and measure recall@5
against the baseline (0.273), changing nothing else.

## Non-goals

LLM-based query understanding (Part 7: later, compared against this rule-based extractor),
doc-type filters (10-K vs 10-Q vs earnings), BM25, reranking.

## Findings that shaped the design (150 FinanceBench questions)

- 25 questions do not contain the catalog company name: they use aliases/tickers
  (`JnJ`, `JPM`, `AMEX`, `Coca Cola`, `MGM`) → an **alias table** is required.
- 128 mention the gold filing's year, 19 mention none, 3 mismatch: forward-looking
  questions (`forecasting for FY2023` → 2022 10-K) and ranges (`FY2023 and Q2 of FY2024`).
  → year filter = **any of the mentioned years**, optionally widened by `year_window`.

## Design

### Filter values (`domain/models.py`)

`Filters = dict[str, MetadataValue | tuple[MetadataValue, ...]]`: a scalar means "equals",
a tuple means "any of". `matches(metadata, filters)` is the single definition, used by the
vector store's pre-filter mask and by the evaluator.

### Extraction (`retrieval/filters.py`)

`FilterExtractor` interface → `CatalogFilterExtractor(company_aliases, year_window)`:

- Text is normalized (lowercase, non-alphanumerics → spaces) so `Coca Cola`, `coca-cola`
  and `Coca-Cola's` all match the alias `coca cola`. Aliases match on word boundaries.
- If aliases of **two different companies** match, no company filter is applied
  (ambiguous; a wrong hard filter is worse than none).
- Years: 4-digit years 1990-2049 not embedded in longer numbers; each widened to
  `[y - window, y + window]`.

Aliases live with the dataset (`datasets/financebench.py: COMPANY_ALIASES`): canonical
name plus common names and tickers, written from general knowledge. In production this
is a company master table. **Caveat:** a few aliases were noticed while reading these
questions, so extraction accuracy here is optimistic; the eval reports it explicitly.

### Retrieval with relaxation (`retrieval/filtered.py`)

`FilteredRetriever(inner, extractor, relax_order=("year", "company"), min_candidates=1)`:
filters = extracted ∪ explicit `query.filters` (explicit wins). If the filtered search
returns fewer than `min_candidates`, drop the next key in `relax_order` and retry
(Part 8 stepwise relaxation; e.g. a company with no filing for the asked year).

### Retriever results carry what was applied

`Retriever.retrieve` now returns `RetrievalResult(candidates, applied_filters, relaxed)`
instead of a bare list, so traces can show which filters were used and relaxed. This
changes `DenseRetriever`, the runner and tests; later stages (fusion, reranking) will add
their diagnostics the same way.

### Evaluation

The runner takes `doc_metadata` (doc_id → metadata, from the indexed chunks) and, per
question, records `applied_filters`, `relaxed`, and `filter_ok` (at least one evidence document
is allowed by the applied filters, so retrieval *could* still find evidence). New status, checked before `not_retrieved`:

| Status | Meaning | Fix |
| --- | --- | --- |
| `filtered_out` | the applied filters excluded every evidence document | extraction / aliases / year policy |

`filter_ok` is also a metric, so it is averaged overall and per segment.

## Config and experiments

```yaml
retrieval:
  top_k: 5
  filters:
    enabled: true
    fields: [company, year]
    year_window: 0
```

- `configs/baseline.yaml`: filters disabled (unchanged behaviour).
- `configs/filters.yaml`: identical except `filters.enabled: true` → the ablation.
- `configs/filters-window1.yaml`: `year_window: 1` → second ablation.

## Test plan

Extractor: aliases, normalization, word boundaries, ambiguity, years and windows, no
matches. `matches` and the store with tuple filters. FilteredRetriever: merge with explicit
filters, relaxation order, no relaxation when results suffice. Runner: `filtered_out`,
`filter_ok`, applied filters in traces. Config validation.
