# HLD 009: Page-aware chunking and contextual chunk headers

## Goal

Backlog 3.2 (+ the simplest form of 3.7). HLD 008 showed the bottleneck is chunk context:
a cash-flow statement split into fragments like "Acquisitions, net of cash acquired 13
(2,023) …" says nothing about *which* statement or *which* years, so neither the embedder
nor a cross-encoder can match it to "FY2018 capital expenditure" (study guide, Part 4:
"heading path prepended"; Part 5: "what exactly gets embedded").

## Findings that shaped the design

- pypdf keeps no fonts, and italic fragments look like headings ("Is a Service
  Contract"), so generic heading detection from plain text is unreliable.
- In 10-K/10-Q filings the **page** is the structural unit that matters most: each
  financial statement and most tables start on a new page with a **title block**, e.g.
  `3M Company and Subsidiaries / Consolidated Statement of Cash Flows / Years ended
  December 31 / (Millions) / 2018 / 2017 / 2016`. Prose pages start with a long
  mid-sentence line and have no title block.
- A prototype title-block extractor finds a title on 158 of 189 evidence pages.

## Design

### Chunk text vs indexed text (`domain/models.py`)

`Chunk` gains `context: str = ""` and `indexed_text` (= `context + "\n\n" + text`, or
`text` if no context). `text` stays exactly `Document.text[char_start:char_end]`, so
offsets, audits and citations are unchanged. **Everything that searches or scores uses
`indexed_text`**: embedding (`IndexStage`), BM25 (`BM25Index.add`) and reranking.

### Page-aware boundaries (`RecursiveChunker(respect_pages=True)`)

The recursive split/merge runs **inside each page** (`Document.page_spans()`), so no
chunk mixes two pages. Fingerprint compatibility: the `respect_pages` key is only added to
the fingerprint when true, so existing chunk sets and indexes keep their IDs.

### Context headers (`ingestion/context.py`)

- `page_title_block(text, max_lines, max_line_chars, max_chars)`: leading lines of a page
  until the first line longer than `max_line_chars` (prose), at most `max_lines`; drops
  "Table of Contents" (also when broken across lines) and bare page numbers; re-attaches
  1-2 letter fragments ("Flow" + "s" → "Flows").
- `ChunkContextualizer` interface; `HeaderContextualizer(document_fields, page_title)`
  builds e.g. `3M | 10k | 2018 | 3M Company and Subsidiaries / Consolidated Statement of
  Cash Flows / Years ended December 31 / (Millions) / 2018 / 2017 / 2016`.
- `ContextualChunker(inner, contextualizer)` decorates any chunker: sets `context`, and
  re-derives chunk IDs and the fingerprint from both, so contextual chunks get their own
  chunk directory and index.

## Config and experiments

```yaml
chunking:
  respect_pages: false
  context:
    enabled: false
    document_fields: [company, doc_type, year]
    page_title: true
```

Each ablation needs its own chunk set and dense index (~4 min to embed):

| Config | vs | Change |
| --- | --- | --- |
| `pages.yaml` | `filters.yaml` | `respect_pages: true` |
| `pages-context.yaml` | `pages.yaml` | + `context.enabled: true` |
| `pages-context-hybrid.yaml` | `pages-context.yaml` | + `mode: hybrid` |
| `pages-context-rerank.yaml` | `pages-context.yaml` | + bge-reranker-base (re-test HLD 008's hypothesis) |

## Test plan

Title block: statement page, prose page, broken "Table of Contents", page numbers vs years,
fragment re-joining, limits. Page spans and page-respecting chunking (no chunk crosses a
page, offsets exact, fingerprint unchanged when off). Contextualizer output, missing
metadata. ContextualChunker IDs/fingerprints. Indexers use `indexed_text`.

## Outcome (2026-11-18)

| Run (all with filters) | recall@5 | nDCG@5 | MRR | recall@10 | evidence in one chunk |
| --- | --- | --- | --- | --- | --- |
| dense, recursive (before) | 0.377 | 0.292 | 0.290 | 0.468 | 62.4% |
| + page boundaries | 0.390 | 0.309 | 0.306 | 0.493 | 72.0% |
| + context headers (**new default**) | 0.427 | 0.339 | 0.339 | 0.547 | 72.0% |
| + hybrid | 0.437 | 0.321 | 0.308 | 0.563 | |
| + bge-reranker-base (instead of hybrid) | 0.343 | 0.252 | 0.252 | 0.503 | |

- Each single step (pages, context, hybrid) is a paired-bootstrap tie: 150 questions
  cannot resolve ~0.03 differences. The cumulative ladder from the Phase 1 baseline
  (0.273 → 0.437) is significant (+0.163, CI [+0.087, +0.243]).
- Segment effects pull in opposite directions: pages help `metrics-generated`
  (0.48 → 0.67), context helps `domain-relevant` (0.18 → 0.36) but pulls
  `metrics-generated` back to 0.50. Untested guess: one shared page header makes all chunks
  of a statement page look alike to the embedder.
- The reranker is significantly worse *with* context: HLD 008's hypothesis is rejected.
- Decision: pages + context (dense) is the new default; hybrid and reranking stay optional.
