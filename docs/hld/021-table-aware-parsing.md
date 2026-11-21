# HLD 021: Table-aware parsing, step 1: keep table rows together (backlog 3.1)

## Goal

FinanceBench answers live in financial tables. The baseline parser (HLD 002, raw pypdf
`extract_text()`) puts **every table cell on its own line**, with lines holding a single
space in between:

```text
Purchases of property, plant and equipment (PP&E)
 
 
(1,577)
 
 
(1,373)
```

A row label and its values end up several lines apart, and the column years appear once
at the top of the page. That causes four problems:

- **The chunker** can split a row from its values.
- **The embedder** sees whitespace noise.
- **The small LLMs** (reranker, generator) have to reassemble the table in their heads.
  HLD 015's commonest wrong answers were exactly this: the wrong column, the wrong table,
  the wrong scale.
- **Token waste:** about half of a table chunk's 400 tokens are blank lines.

Study guide Part 3: *"tables flattened into text lose their structure; keep rows together
and keep headers with the values."*

## Step 1 (this HLD): layout mode + normalisation, no new dependency

pypdf's `extract_text(extraction_mode="layout")` positions text the way it appears on the
page, so a visual row stays on one line. `normalize_layout_text()` then compacts it:

| Rule | Before | After |
| --- | --- | --- |
| a column gap (3+ spaces) becomes ` \| ` | `Purchases of PP&E        (1,577)       (1,373)` | `Purchases of PP&E \| (1,577) \| (1,373)` |
| a currency sign joins its number | `$      5,363` | `$5,363` |
| a percent sign joins its number (never before a word) | `37.2   %` | `37.2%`; `2017 \| % change` is unchanged |
| non-breaking spaces become spaces; blank runs collapse; lines are stripped | | |

Hyphens at line ends are **kept**. In these filings they are mostly real compounds split
by the layout (`local-` / `currency`); re-joining them produced "localcurrency".

**What step 1 does not do:** it doesn't repeat column headers on each row (a chunk that starts
mid-table still lacks "2018 | 2017"), and it doesn't detect table boundaries. That's step 2,
if step 1 earns its place.

## Design

- **`PypdfLayoutParser(PypdfParser)`** (`parser: pypdf-layout`) overrides one method,
  `_page_text`. Failure handling, page alignment and metadata are inherited.
- **Artifacts never mix.** Before this, parsed documents always went to
  `data/processed/parsed/`, and chunk folders and IDs were keyed by the chunker settings
  only. Now:
  - parsed documents go to `parsed_dir_for(parser)`: the baseline keeps `parsed/`, and
    others get `parsed-<parser>/`;
  - the parser joins the chunker fingerprint (`source`), so new chunk IDs and a new index
    are built, but only for non-baseline parsers.

  Every existing pypdf artifact, index and cache (plans, grades, judge verdicts) keeps its
  path and IDs. This was verified: the committed configs still find their indexes.
- **Configs:** `planner-rules-layout.yaml` and `planner-llm-rerank-llm-layout.yaml`. Each
  differs from its base **only** in `parsing.parser` (a config test enforces it).

## Evaluation plan

1. **Parse audit** (rung 1): evidence-token coverage per page, baseline vs layout. Layout
   must not *lose* text.
2. **Chunk audit:** does the evidence still fit in one chunk? Chunk count and tokens per
   chunk.
3. **Retrieval, default config** (`planner-rules`, no LLM): dev recall@5, paired.
4. **Retrieval, best config** (LLM planner + LLM reranker): new chunks mean new grades,
   so this is a cold run (~45 min).
5. **Generation** (llama3.1:8b on the best retrieval), if retrieval holds. This is where
   table structure should matter most: numeric accuracy and the ladder's "in prompt but
   wrong".

## Results

### A guard found by the parse audit: never lose text

The first layout run returned an **empty page** for AMEX's 10-K cover page (coverage
0.00). Across the corpus, layout mode returned nothing on 14 of 11,948 pages (mostly
cover pages) and fewer than 80% of the plain-extraction words on 17 more.
`PypdfLayoutParser` now falls back to plain text for such a page (`min_word_ratio` 0.8),
and counts it in `metadata["layout_fallback_pages"]`. All results below include the guard.

### Rungs 1-2: parse and chunk audits (all 189 evidence spans)

| | pypdf | pypdf-layout |
| --- | --- | --- |
| parse audit pass rate (coverage ≥ 0.8) / mean coverage | 96.8% / 0.985 | **97.4%** / 0.985 |
| chunks (400 tokens, page-bounded, context headers) | 31,259 | 31,932 |
| chunk tokens p50 | 374 | 328 |
| **evidence fitting in one chunk** | **72.0%** | 67.7% |
| parse time | — | 474 s (both extraction modes, ~39 ms/page) |

Layout text keeps all the evidence, but slightly less of it fits in a single chunk. The
`|` separators are tokens, so a dense table page needs a few more chunks.

### Retrieval (dev, n=96, paired)

| Config | pypdf | layout | Δ (95% CI) |
| --- | --- | --- | --- |
| default (`planner-rules`) recall@5 | 0.484 | 0.490 | +0.006: tie |
| best (LLM planner + LLM reranker, fresh grades) recall@5 | 0.740 | 0.745 | +0.005 [−0.073, +0.083], 12/11: tie |
| best, MRR | 0.628 | 0.590 | −0.037 [−0.093, +0.018]: tie |

### Generation (llama3.1:8b on the best retrieval, dev, n=96)

| | pypdf | layout |
| --- | --- | --- |
| numeric correct | **10/32** | 8/31 |
| paired numeric: base ✓ → layout ✗ / base ✗ → layout ✓ | | **5 / 3** (not significant) |
| evidence in prompt | 0.79 | 0.78 |
| errors / refused | 4 / 16 | 4 / 19 |
| over-refusal | 0.11 | 0.12 |
| input tokens (all 96) | 188k | 184k |

**Reading the flipped answers shows why there's no gain.** With layout text, the model
usually *read the right figure*, then reasoned wrongly:

- Costco total assets: the working lists "$59,268 [S1]" correctly, then subtracts FY2020
  unasked and answers $3,712.
- PepsiCo EBITDA: it picks net income instead of operating income.

Those are reasoning errors by the 8B model, not table-reading errors. HLD 015 already
noted that this generator's answers are unstable when the context changes.

## Decision

- **Not adopted as the default.** Step 1 doesn't measurably help retrieval (ties on both
  configs) or generation (5 vs 3 flips), and it slightly lowers single-chunk evidence fit.
  `pypdf` stays the default parser.
- **Kept as an opt-in parser** (`parser: pypdf-layout`, with three `*-layout.yaml` configs),
  because:
  - its text is clearly more readable for humans and stronger models (rows and years
    together);
  - the per-parser artifact separation is a lasting improvement in its own right.
- **Step 2 (repeat column headers on each row, detect table boundaries) is deferred.** The
  measured bottleneck is the generator's reasoning, not table reading. Better table text
  is worth re-testing with a stronger generator (the parked gateway comparison) or with
  the judges, not with more parsing work against this 8B model.

## Lessons

1. **The audit caught a silent loss before any retrieval number moved:** empty cover
   pages. Every parser change needs the parse audit first.
2. **"Obviously better" text isn't automatically better retrieval.** The embedder and the
   LLM grader found the right pages just as often from the flattened tables.
3. **Read the failures, not only the totals.** The flipped answers show the right
   numbers being read and then misused. That points at the generator, not the parser.

