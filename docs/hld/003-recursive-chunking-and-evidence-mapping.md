# HLD 003: Recursive chunking and evidence → chunk mapping

## Goal

Backlog 1.4 + 1.2. Split parsed documents into ~400-token chunks with **exact source
offsets**, store them as artifacts, and define how a retrieved chunk is judged relevant
to a golden question, so the retrieval metrics (HLD 001) can score real rankings.
Add a **chunk audit**: does each piece of evidence land inside one chunk, or is it cut
across several? (study guide, Part 4; debugging-ladder rung 1 for chunking).

## Non-goals

Structural/heading-aware, semantic, parent-child or contextual chunking (backlog 3.2,
3.7). Cleaning (3.1). Embedding (1.5).

## Design

### Token counting

`TokenCounter` interface; `TiktokenCounter("cl100k_base")` as the baseline. The
embedding model is not chosen yet (1.5); when it is, its own tokenizer may replace this
and the indexer must verify every chunk fits the model's max sequence length.

### `RecursiveChunker(Chunker)`

1. **Split** `[0, len(text))` with separators `["\n\n", "\n", ". ", " "]`, keeping each
   separator attached to the piece on its left, so the pieces tile the text exactly. A
   piece still over `chunk_size` tokens is split with the next separator; past the last
   one it is cut into fixed character windows.
2. **Merge** adjacent pieces greedily while the summed token count stays ≤ `chunk_size`.
   On a new chunk, carry trailing pieces worth ≤ `chunk_overlap` tokens (Part 4 overlap).
3. **Trim** surrounding whitespace by moving offsets (never editing text); drop empty chunks.

Token counts are the **sum of piece counts** (BPE is near-additive; recounting every
candidate would be quadratic). `metadata.token_count` stores the exact count of the final text.

Each chunk: `chunk_id = make_chunk_id(doc_id, start, end, fingerprint)`, `page_start/end`
from `Document.page_index_at`, and the document's metadata (company, year, ...) copied so
filters work at chunk level. Chunks may span pages: the baseline ignores structure on purpose.

`fingerprint` = hash of `{strategy, chunk_size, chunk_overlap, tokenizer, separators}`.

### Artifacts

`ChunkSet(doc_id, chunker_fingerprint, chunks)` per document at
`data/processed/chunks/<fingerprint>/<doc_id>.json`. The fingerprint directory lets several
chunk configurations coexist for ablations. `ChunkStage` mirrors `ParseStage` (idempotent,
failure-tolerant, report).

### Evidence mapping (relevance units)

Relevance unit = evidence page: `"<doc_id>#p<page_index>"`. For one golden item:

- `qrels(item)` = `{unit: 1}` for each distinct evidence page.
- `relevance_ids(ranked_chunks, item)`: each chunk becomes the unit of the first evidence
  page it covers (`doc_id` matches and `page_start ≤ page ≤ page_end`), else its own
  `chunk_id` (non-relevant).

This is **page-level relevance**, the granularity FinanceBench labels at. It is lenient:
a chunk holding the page's footer counts. The chunk audit quantifies how often the
evidence actually fits in one chunk; a stricter span-level relevance is a later option.

### Chunk audit

For each evidence span: chunks overlapping its page, and the **best single-chunk
coverage** (`token_coverage(evidence, chunk.text)`, from HLD 002). Summary: share of
evidence whose best chunk ≥ threshold (`evaluation.parse_audit_min_coverage`), plus the
chunk count and token-size distribution.

## Config

```yaml
chunking:
  strategy: recursive
  chunk_size: 400
  chunk_overlap: 0
  tokenizer: cl100k_base
```

## Alternatives

Count words instead of tokens (no dependency, but the embedder's limit is in tokens).
LangChain's splitter (same algorithm, but loses offsets).

## Test plan

Fake whitespace token counter: chunks tile the text with exact offsets, respect size,
prefer paragraph boundaries, apply overlap, hard-split unbroken text, trim whitespace,
deterministic IDs, page spans across page boundaries, metadata copied. Mapping and audit
with hand-built chunks. One `slow` test for the real tiktoken counter.
