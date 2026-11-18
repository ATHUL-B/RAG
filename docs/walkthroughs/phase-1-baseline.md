# Phase 1 walkthrough: the baseline retrieval pipeline

This document explains **what Phase 1 built, why, and how the code is organised**, so you
can read the code with its intention in mind. Read it top to bottom once; afterwards use
§12 ("Reading order") as a map.

Study-guide references look like **(Part 9)** and point to
`RAG from Basics to Advanced A Production-Scale Study Guide.md`.

---

## 1. TL;DR

Phase 1 built the **simplest complete RAG retrieval pipeline**, plus the measuring
equipment around it:

```
FinanceBench PDFs ─► parse ─► chunk ─► embed ─► exact vector index ─► top-5 chunks
       │                │        │                                        │
   golden set      parse audit  chunk audit                      retrieval eval + traces
```

It is deliberately naive (no cleaning, no filters, no BM25, no reranker, no LLM yet).
That is the point: **a baseline you can measure**, so every later improvement in Phase 3
is one change with a before/after number (the "ablation ladder", Part 13).

Baseline results on 150 FinanceBench questions:

| Where evidence survives | Result |
| --- | --- |
| After parsing (evidence text present on its page) | **96.8%** of evidence spans |
| After chunking (evidence fits inside ONE chunk) | **62.4%** |
| After retrieval (evidence page in the top 5) | **recall@5 = 0.273**, hit@5 = 0.300 |

The biggest single reason for misses: **40.7% of all top-5 slots are filled by a
different filing of the same company** (e.g. 3M 2022 instead of 3M 2018). That is the
"version trap" of the guide's ZenithCare example (Part 8), now measured on real data.

---

## 2. What Phase 1 set out to do, and why in this order

The study guide's learning path (Part 15) starts with: *"Take some PDFs, chunk at 400
tokens, embed, retrieve top-5, measure recall@5."* We followed it, with one rule from
`CLAUDE.md`: **measure first, then improve**.

The order of work was chosen so that every step can be checked by the one after it:

| Step | Backlog | What | Why at this point |
| --- | --- | --- | --- |
| 0 | 0.1-0.2 | Skeleton, domain model, interfaces, dataset adapter | Everything else plugs into these |
| 1 | 1.1 | Retrieval metrics | A wrong metric silently invalidates every experiment, so it is built and verified *first*, against the guide's hand-computed examples |
| 2 | 1.3 | PDF parsing + **parse audit** | Debugging-ladder rung 1: is the evidence even in the extracted text? |
| 3 | 1.4 + 1.2 | Chunking + **chunk audit** + evidence→chunk mapping | Rung 1 again: does the evidence survive being cut into chunks? And how do we score chunk rankings against page-level labels? |
| 4 | 1.5 | Embeddings + exact vector index + dense retriever | The retrieval itself |
| 5 | 1.6-1.7 | Eval runner, traces, failure ladder, first baseline | Rung 2: when retrieval misses, *why*? |

The **debugging ladder** (Part 13) is the idea that ties it together. When an answer is
wrong, walk down the pipeline and find the *first* broken stage:

1. Is the evidence in the index? → parsing / chunking (our two audits)
2. Was it in the first-stage candidates? → query, filters, embeddings, hybrid (eval runner status)
3. Did it survive reranking? → (Phase 3)
4. Was it in the final prompt? → (Phase 2)
5. Did the model use it? → (Phase 2)

Phase 1 instruments rungs 1 and 2.

---

## 3. The dataset: FinanceBench

Chosen in `docs/adr/0001-dataset.md`. Why it fits the guide:

- **Real, messy PDFs**: SEC 10-K/10-Q filings with tables (Part 3 problems are real).
- **Same company across years**: 3M 2015…2023. Near-identical wording year to year is
  exactly the guide's "2022 vs 2024 policy wording" trap.
- **Labelled evidence**: every question names the filing and the **zero-based page**
  holding the evidence, plus the annotator's evidence text. That lets us measure
  retrieval separately from answer generation.

Numbers: 150 open-source questions (3 types × 50), referencing **84** filings; the
catalog describes 361 filings; we downloaded the 84 referenced PDFs (158 MB).

A golden question, as our code sees it (`GoldenItem`):

```
item_id:          financebench_id_03029
question:         What is the FY2018 capital expenditure amount (in USD millions) for 3M? ...
reference_answer: $1577.00
evidence:         (SourceSpan(doc_id="3M_2018_10K", page_index=59, text="...cash flow statement..."),)
tags:             {type:metrics-generated, reasoning:information-extraction, evidence:single}
```

**Tags** (`<dimension>:<value>`) let every report be broken down per segment, because an
overall average can hide a collapse on one kind of question (Part 13).

---

## 4. Repository layout

```
src/rag/                    the library (all logic lives here; scripts are thin)
  domain/                   plain immutable data shared by every stage
    models.py                 Document, Page, Chunk, ChunkSet, SourceSpan, ScoredChunk,
                              Query, Answer, GoldenItem
    ids.py                    deterministic IDs and config fingerprints
  ingestion/                OFFLINE: raw files → chunks          (Parts 3-4)
    base.py                   interfaces: Parser, Cleaner, MetadataEnricher, Chunker
    pdf_parser.py             PypdfParser
    enrichers.py              StaticMetadataEnricher (company/year/... from the catalog)
    parse_stage.py            ParseStage: parse → enrich → save, for many documents
    tokenization.py           TokenCounter interface, TiktokenCounter
    chunkers.py               RecursiveChunker
    chunk_stage.py            ChunkStage: load documents → chunk → save
    factory.py                config → Parser / Chunker
  indexing/                 OFFLINE: chunks → vectors → index    (Part 5)
    base.py                   interfaces: Embedder, VectorStore, KeywordIndex
    embedders.py              SentenceTransformerEmbedder (bge-small)
    vector_store.py           ExactVectorStore (NumPy, filters, save/load)
    index_stage.py            IndexStage: embed every ChunkSet into a store
    factory.py                config → Embedder; where an index lives on disk
  retrieval/                ONLINE: question → ranked chunks     (Parts 6-10)
    base.py                   interfaces: QueryRewriter, Retriever, Fuser, Reranker
    dense.py                  DenseRetriever
  generation/               ONLINE: chunks → answer             (Part 11, Phase 2)
    base.py                   interfaces only so far
  evaluation/               measuring                            (Part 13)
    retrieval_metrics.py      recall/precision/hit@K, reciprocal rank, nDCG (pure functions)
    retrieval_report.py       per-query scoring, aggregation per segment
    evidence.py               golden evidence → "relevance units" → qrels
    parse_audit.py            did evidence survive parsing?
    chunk_audit.py            does evidence fit in one chunk?
    retrieval_runner.py       run the golden set through a retriever; traces; failure status
    base.py                   Judge interface (Phase 2)
  datasets/
    http.py                   Fetcher interface, UrllibFetcher (timeouts, retries)
    financebench.py           downloader + reader + mapping to GoldenItem
  observability/
    timing.py                 StageTimer (per-stage milliseconds, injectable clock)
    stats.py                  latency_summary (p50/p95/p99/max)
    structured_logging.py     JSON log lines
  config/
    settings.py               typed config models (AppConfig and sections)
    loader.py                 YAML → validated AppConfig
  storage.py                atomic writes, safe keys, JsonArtifactStore
  errors.py                 RagError and one subclass per stage

scripts/                    command-line entry points, one per pipeline step (§9)
configs/baseline.yaml       the one experiment defined so far
tests/                      mirrors src/; tests/support holds shared fakes
docs/                       architecture, backlog, ADRs (decisions), HLDs (designs), this doc
experiments/LOG.md          every measured result, in order
evals/                      our own golden questions (Phase 2+)
data/   (gitignored)        raw/ downloads, processed/ artifacts, indexes/
runs/   (gitignored)        audit reports and evaluation runs
```

**Rule of thumb:** `src/rag` contains every decision; `scripts/` only wires objects
together from config and prints results. If you want to know *how* something works, read
`src/`; if you want to know *how to run* it, read `scripts/`.

---

## 5. Design ideas that run through all the code

These are the "why does the code look like this?" answers. Most come from `CLAUDE.md`.

### 5.1 One interface per pipeline stage

Each stage is an abstract base class (`ingestion/base.py`, `indexing/base.py`,
`retrieval/base.py`, `generation/base.py`):

```python
class Chunker(ABC):
    @property
    @abstractmethod
    def fingerprint(self) -> str: ...
    @abstractmethod
    def chunk(self, document: Document) -> Sequence[Chunk]: ...
```

Concrete classes (`RecursiveChunker`, `PypdfParser`, `DenseRetriever`, …) implement them,
and a **factory** builds the right one from config (`ingestion/factory.py`,
`indexing/factory.py`). Consequence: a Phase 3 experiment such as "structural chunking"
is a *new class + a config value*, and nothing else changes. That is what makes the
ablation ladder cheap.

### 5.2 Dependency injection: collaborators are passed in

Classes receive what they depend on instead of creating it:

```python
DenseRetriever(embedder, store)
ParseStage(parser, enricher, store, clock=...)
RetrievalEvalRunner(retriever, evaluator, indexed, top_k, clock=...)
```

So tests pass in **fakes** (a hashing embedder, a word-counting tokenizer, a ticking
clock) and run in milliseconds, offline, deterministically. Real models are used only in
tests marked `slow`.

### 5.3 Immutable domain models

All data objects in `domain/models.py` are pydantic models with `frozen=True` and
`extra="forbid"`. A stage never edits its input; it returns a new object
(`document.model_copy(update=...)`). Unknown fields are errors, so schema drift is caught
immediately. Validators enforce invariants (pages ascending, `char_end > char_start`,
chunks belong to their `ChunkSet`'s document).

### 5.4 An inspectable artifact after every offline stage

| Stage | Artifact | Model |
| --- | --- | --- |
| Download | `data/raw/financebench/…jsonl`, `pdfs/*.pdf` | upstream files |
| Parse | `data/processed/parsed/<doc_id>.json` | `Document` |
| Chunk | `data/processed/chunks/<chunker_fp>/<doc_id>.json` | `ChunkSet` |
| Index | `data/indexes/dense/<chunker_fp>__<model>/` | `vectors.npy`, `chunks.jsonl`, `manifest.json` |
| Eval | `runs/<run_id>/traces.jsonl`, `metrics.json` | `QueryTrace`, run summary |

You can open any of these and read exactly what the next stage saw. This is what makes
"is the problem in ingestion or retrieval?" answerable with evidence instead of guesses.
Writes are **atomic** (write `*.part`, then rename; `storage.atomic_write_bytes`), so a
crash never leaves a half-written file that looks valid.

### 5.5 Deterministic IDs and fingerprints

- `config_fingerprint(dict)` = first 16 hex chars of SHA-256 of the canonical JSON.
- `chunk_id = hash(doc_id, char_start, char_end, chunker_fingerprint)`.

Same input → same ID on any machine. Changing the chunker settings changes the
fingerprint, so new chunks land in a **new directory** and new IDs; old and new
configurations can coexist for comparison, and a stale index can never be reused by
accident (the index directory is keyed by chunker fingerprint *and* model).

### 5.6 Evidence is labelled at the source, not at the chunk

Golden evidence is a `SourceSpan(doc_id, page_index, text)`, independent of how we chunk.
Evidence is mapped to chunks **at evaluation time** (§7.8). If we labelled chunk IDs,
every re-chunking would invalidate the golden set.

### 5.7 Config, not constants

Every tunable number (chunk size, overlap, K, model, thresholds) lives in
`configs/baseline.yaml`, validated by `config/settings.py`. A typo in a key is an error
(`extra="forbid"`), not a silently ignored setting. An experiment is fully described by
**git SHA + config file + dataset version**.

### 5.8 Errors say which stage failed

`errors.py`: `RagError` → `ConfigError`, `DatasetError`, `StorageError`,
`IngestionError`, `IndexingError`, `RetrievalError`, `GenerationError`. Batch stages
(`ParseStage`, `ChunkStage`, the downloader) catch their stage's error **per item**,
record it in a report, and continue, so one bad PDF doesn't kill a 3-minute run.

---

## 6. The domain model, one object at a time

All in `src/rag/domain/models.py`.

| Model | Meaning | Key fields / behaviour |
| --- | --- | --- |
| `Page` | One PDF page's text | `page_index` (zero-based), `text` |
| `Document` | A parsed file | `doc_id`, `source_uri`, `pages`, `metadata`. `.text` joins pages with `"\n\n"`; `.page_index_at(offset)` maps a character offset back to its page |
| `Chunk` | The unit that is embedded, retrieved and (later) cited | `chunk_id`, `doc_id`, `text`, `char_start/char_end` into `Document.text`, `page_start/page_end`, `metadata` (company, year, token_count, chunker…) |
| `ChunkSet` | All chunks of one document from one chunker | the chunk artifact |
| `SourceSpan` | Where evidence lives in the source | `doc_id`, `page_index`, `text` |
| `GoldenItem` | One evaluation question | `question`, `reference_answer`, `evidence`, `tags`; `.answerable` = has evidence |
| `Query` | A search request | `text`, `filters` (e.g. `{"company": "3M", "year": 2018}`) |
| `ScoredChunk` | A candidate from some stage | `chunk`, `score`, `rank` (1-based), `stage` ("dense", later "bm25", "rrf", "rerank") |
| `Answer` | Generated answer (Phase 2) | `text`, `cited_chunk_ids`, `refused` |

Conventions: pages are **zero-based** everywhere (matches pypdf *and* FinanceBench, which
we verified on 50/50 evidence pages); offsets are half-open `[start, end)`.

---

## 7. The pipeline, stage by stage

Each subsection: **goal → code → how it works → design decisions → result**.

### 7.1 Download (`datasets/http.py`, `datasets/financebench.py`)

**Goal:** get FinanceBench onto disk safely and repeatably.

- `Fetcher` is an interface; `UrllibFetcher` implements it with a timeout, HTTPS-only,
  and retries with exponential backoff (1 s, 2 s, 4 s) on 429/5xx and network errors,
  but **not** on 404 (retrying a missing file is pointless).
- `FinanceBenchDownloader` is **idempotent** (skips existing files), writes atomically,
  checks a PDF really starts with `%PDF` (a server error page saved as `.pdf` would
  poison parsing), and rejects document names that could escape the target folder
  (`../`), because names come from downloaded data.
- `FinanceBenchDataset` reads the JSONL files into validated records
  (`FinanceBenchQuestion`, `FinanceBenchDocument`) and maps them to domain objects:
  `golden_items()`, `referenced_doc_names()`, `document_metadata()` (company, sector,
  doc_type, year: the fields filters will use in Phase 3).

Run: `make download-financebench` (metadata + the 84 referenced PDFs).

### 7.2 Parsing (`ingestion/pdf_parser.py`, `enrichers.py`, `parse_stage.py`)

**Goal:** PDF → `Document` with one `Page` per PDF page. (Part 3)

- `PypdfParser.parse(path, doc_id)` calls pypdf's `extract_text()` per page.
  - A page that fails to extract becomes an **empty page** (counted in
    `metadata["failed_pages"]`) instead of being dropped: dropping it would shift every
    later page index and break the evidence labels.
  - An unreadable file raises `IngestionError`.
  - Some filings are **AES-encrypted with an empty password**; pypdf needs the
    `cryptography` package for them, hence the `pypdf[crypto]` dependency. (We found this
    when the first probe crashed.)
  - Records `metadata["parser"] = "pypdf-<version>"`, so artifacts say what made them.
- `StaticMetadataEnricher` merges catalog metadata (company, sector, doc_type, year).
  A document missing from the catalog, or with a conflicting value, is an **error**: a
  chunk without `year` would silently escape every year filter later.
- `ParseStage.run(sources, force)` does parse → enrich → save per document, skips
  already-parsed ones unless `--force`, records failures, and times each document.

The baseline is **deliberately naive**: no cleaning, no table handling. Better parsers
are Phase 3 (3.1) and will be compared with the same audit.

Result: 84/84 filings, 12,013 pages, 0 failures, 17.4 ms/page.

### 7.3 Parse audit (`evaluation/parse_audit.py`)

**Goal:** rung 1. Did each piece of golden evidence survive extraction?

For each evidence span: `token_coverage(evidence_text, parsed_page_text)` = share of the
evidence's tokens (lowercase alphanumeric runs) that appear anywhere on the parsed page.
≥ 0.8 → `passed`. It also checks the neighbouring pages and reports if one matches
better (a page-offset hint), and reports "document not parsed" separately from "0%".

Result: **96.8% pass, mean coverage 0.985.** We inspected the worst cases rather than
trusting the number: the text *was* extracted; the annotators' evidence text has its
spaces stripped (`Cashandcashequivalents`), which deflates the score. So 96.8% is a lower
bound. The inspection also revealed real noise for later cleaning: non-breaking spaces
(`\xa0`), words broken across lines (`sha\nre`), and table rows flattened into one line.

### 7.4 Chunking (`ingestion/tokenization.py`, `chunkers.py`, `chunk_stage.py`)

**Goal:** split each document into ~400-token units with exact offsets. (Part 4)

`RecursiveChunker` in three phases:

1. **Split.** Try the coarsest separator first: `"\n\n"` (paragraphs and page breaks),
   then `"\n"`, `". "`, `" "`. Each separator stays attached to the piece on its left,
   so the pieces tile the text exactly. A piece still over 400 tokens is split with the
   next separator; past the last one it is cut into fixed character windows.
2. **Merge.** Pack adjacent pieces greedily while the token sum stays ≤ 400. With
   `chunk_overlap > 0`, the next chunk starts with the previous chunk's trailing pieces
   worth ≤ overlap tokens (the baseline uses 0).
3. **Trim.** Move offsets past surrounding whitespace (the text is never edited), drop
   empty chunks.

Each chunk gets its ID, page range (via `Document.page_index_at`), the document's
metadata, `chunker` fingerprint and exact `token_count`. Chunks **may span pages**: the
baseline ignores document structure on purpose (structural chunking is Phase 3, 3.2).

Tokens are counted with tiktoken `cl100k_base` behind a `TokenCounter` interface; tests
use a word counter instead.

Result: 26,264 chunks in 6.4 s; tokens p50 385, p95 398, max 401.

A real chunk (text shortened). Notice how much of it is table whitespace from parsing:

```json
{
  "chunk_id": "a6296da578a1a1fa",
  "doc_id": "3M_2018_10K",
  "text": "1\n \n \n(15)\n \nStock-based compensation\n \n \n291\n \n \n291\n \n \n ...",
  "char_start": 218497, "char_end": 219596,
  "page_start": 58, "page_end": 59,
  "metadata": {"company": "3M", "year": 2018, "doc_type": "10k", "sector": "Industrials",
               "chunker": "a02d9252b7b267ae", "token_count": 330, "parser": "pypdf-7.1.0", ...}
}
```

### 7.5 Chunk audit (`evaluation/chunk_audit.py`)

**Goal:** rung 1 again. Does each piece of evidence fit inside **one** chunk, or was it
cut across several?

For each evidence span: take the chunks covering its page and report the **best single
chunk's** token coverage.

Result: **62.4%** of evidence fits in one chunk (vs 96.8% surviving parsing). Split by
evidence length, this is two different problems:

| Evidence length | Fits in one chunk | Meaning |
| --- | --- | --- |
| ≤ 400 tokens (92 spans) | 90.2% | the 9 misses are unlucky chunk boundaries |
| > 400 tokens (97 spans) | 36% | mostly whole financial statements; no 400-token chunk can hold one |

This is a lesson measured *before any retrieval*: for half the evidence, a 400-token
recursive chunker cannot produce a chunk that contains it. Fixes are structural/table
chunking, parent-child retrieval, or bigger chunks (Phase 3).

### 7.6 Embedding and indexing (`indexing/embedders.py`, `vector_store.py`, `index_stage.py`)

**Goal:** turn every chunk into a vector and store them for search. (Part 5)
Decisions in `docs/adr/0003-embedding-model-and-vector-store.md`.

- **`SentenceTransformerEmbedder`** wraps `BAAI/bge-small-en-v1.5`:
  - pinned model **revision**, so the same config always means the same weights;
  - output is **L2-normalized**, so cosine similarity = dot product;
  - queries get bge's instruction prefix
    `"Represent this sentence for searching relevant passages: "`, passages don't
    (this is why `embed_query` and `embed_documents` are separate methods);
  - the model is loaded **lazily**, so creating the object (e.g. to compute an index
    path) is cheap; `device: auto` picks Apple MPS, then CUDA, then CPU.
- **`ExactVectorStore`**: one `float32` matrix (26,264 × 384) plus the chunk list.
  - `search` = optional **metadata pre-filter mask** → matrix × query vector → stable
    sort → top-k. Filtering *before* ranking matters: with post-filtering, wrong-year
    chunks would take the top-k slots and then be thrown away (Part 8).
  - Exact search is the **ground truth** we will later use to measure how much an
    approximate index (HNSW, Phase 4) loses ("index recall").
  - `save`/`load` with a `manifest.json` (`model_id`, `dimension`, `count`). Loading
    with a different model is refused: vectors from two models are not comparable.
  - Rejects wrong shapes, unnormalized vectors and duplicate IDs in a batch.
- **`IndexStage`** embeds all chunk sets in batches and reports throughput and how many
  chunks exceed the model's 512-token limit (those are truncated by the model).

Result: 26,264 chunks in 227 s (116 chunks/s on MPS), 87 MB on disk.
**91 chunks are truncated**: they are ≤ 401 tiktoken tokens but > 512 bge tokens,
because bge's tokenizer splits numbers more finely. A real-world reminder that the
chunker and the embedder must agree on what a "token" is (backlog note).

### 7.7 Dense retrieval (`retrieval/dense.py`)

**Goal:** question → top-K chunks. (Part 6)

```python
vector = embedder.embed_query(query.text)
return store.search(vector, k, query.filters or None)
```

That is the whole "naive RAG" retriever. Latency: **9.3 ms p50, 30.2 ms p95** per
question (embedding the query + 26k dot products).

### 7.8 Evaluation (`evaluation/evidence.py`, `retrieval_metrics.py`, `retrieval_report.py`, `retrieval_runner.py`)

**Goal:** score the retriever against the golden set, and explain each miss.

**Relevance units (`evidence.py`).** FinanceBench labels evidence by *page*, but we
retrieve *chunks*. The bridge: an evidence page is a unit `"<doc_id>#p<page>"`.

- `qrels_for(item)` → `{"3M_2018_10K#p59": 1}` (one entry per distinct evidence page).
- `relevance_ids(chunks, item)` turns a chunk ranking into unit IDs: a chunk covering an
  evidence page (same doc, page inside its range) becomes that unit; any other chunk
  keeps its own `chunk_id` (never relevant). This is **page-level relevance**: lenient by
  design (a chunk with only the page's footer counts); the chunk audit shows how lenient.

**Metrics (`retrieval_metrics.py`).** Pure functions of `(ranked_ids, qrels)`:

| Metric | Question it answers |
| --- | --- |
| `recall@K` | What share of the evidence pages reached the top K? (repeats count once) |
| `precision@K` | What share of the top K is evidence? (each relevant position counts) |
| `hit@K` | Did at least one evidence page reach the top K? |
| `mrr` | 1 / rank of the first evidence (averaged): how far down must you look? |
| `ndcg@K` | Graded; rewards putting the best evidence first; missing evidence lowers it |

Repeated IDs are allowed because two chunks from one evidence page map to the same unit.
These were verified against the guide's worked examples (recall@5 = 1/3, nDCG@4 = 0.68…)
*before* anything else used them (`tests/evaluation/test_retrieval_metrics.py`).

**Aggregation (`retrieval_report.py`).** `RetrievalEvaluator(ks=[1,3,5,10])` scores one
query at every K; `aggregate()` averages overall **and per tag**, with counts.

**The runner (`retrieval_runner.py`).** For each answerable golden question:

1. `Query(text=question)`: no filters, no rewriting (naive baseline).
2. Retrieve `max(ks, top_k)` = 10 candidates, timed.
3. Compute metrics + `doc_hit@5` (did *any* top-5 chunk come from the right filing?).
   `doc_hit` separates "wrong filing" from "right filing, wrong page".
4. Assign a **status** at `top_k = 5`:

| Status | Meaning | Ladder rung → where to fix |
| --- | --- | --- |
| `not_in_index` | no chunk covers any evidence page | 1 → parsing/chunking |
| `not_retrieved` | evidence indexed, none in top 5 | 2 → query, filters, embeddings, hybrid |
| `partial` | some evidence pages in top 5, not all | 2 (multi-evidence) |
| `hit` | all evidence pages in top 5 | - |

5. Write a **trace**: everything needed to debug the question without re-running it.

A real trace (shortened):

```json
{
  "item_id": "financebench_id_03029",
  "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? ...",
  "evidence_units": ["3M_2018_10K#p59"],
  "status": "not_retrieved",
  "retrieved": [
    {"rank": 1, "doc_id": "3M_2023Q2_10Q", "page_start": 71, "score": 0.772, "relevant": false},
    {"rank": 2, "doc_id": "3M_2022_10K",   "page_start": 35, "score": 0.767, "relevant": false}
  ],
  "timings_ms": {"retrieve": 48.5},
  "metrics": {"recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "doc_hit@5": 1.0}
}
```

Read it as: the evidence is page 59 of the 2018 10-K; the retriever's best guesses are
3M's **2023** 10-Q and **2022** 10-K: the version trap.

---

## 8. Results, and what they teach

From `experiments/LOG.md` (run `baseline-20261118T085010Z`):

| | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| recall | 0.103 | 0.217 | **0.273** | 0.357 |
| hit | 0.113 | 0.233 | 0.300 | 0.393 |

- Status at top 5: 37 hit, 8 partial, **105 not_retrieved, 0 not_in_index** → every miss
  is a rung-2 (retrieval) failure; nothing is missing from the index.
- `doc_hit@5` = 0.813: the right filing usually appears, but not the right page.
- Weakest segments: `reasoning:logical-reasoning` recall@5 0.029, `evidence:multi` 0.143.

What fills the 750 top-5 slots (150 questions × 5):

| | Share |
| --- | --- |
| Same company, **other filing** (version trap) | **40.7%** |
| Right filing, wrong page | 29.9% |
| Different company | 22.1% |
| The evidence page | 7.3% |

**Lessons:**

1. Parsing is not the bottleneck (96.8%); chunking loses more (62.4%); retrieval loses
   most (27.3%). You only know this because each stage was measured separately.
2. Embeddings capture *topic*, not *which year*. Ten 3M filings all talk about capital
   expenditure in near-identical words. The fix is not a better embedding model but a
   **metadata filter** (Part 8): take the company and year from the question and search
   only that slice. That is why Phase 3 starts with backlog 3.3.
3. Within the right filing, the right page is still hard (29.9%): table-heavy chunks
   full of whitespace and numbers embed poorly. That points at cleaning (3.1),
   structural chunking (3.2), BM25 for exact terms and numbers (3.4), and reranking (3.5).

---

## 9. Running everything

From the repo root (`uv` must be on your PATH, see README):

```bash
make install                                   # create .venv
make download-financebench                     # data/raw/financebench (84 PDFs)
uv run python scripts/parse_documents.py       # → data/processed/parsed/        (~3.5 min)
uv run python scripts/audit_parsing.py         # → runs/parse_audit/baseline.json
uv run python scripts/chunk_documents.py       # → data/processed/chunks/<fp>/   (~6 s)
uv run python scripts/audit_chunking.py        # → runs/chunk_audit/baseline.json
uv run python scripts/build_index.py           # → data/indexes/dense/<fp>__<model>/ (~4 min)
uv run python scripts/evaluate_retrieval.py    # → runs/<run_id>/traces.jsonl, metrics.json
make check                                     # lint, types, tests (what CI runs)
```

Every script takes `--config` (default `configs/baseline.yaml`) and is safe to re-run:
existing artifacts are skipped (use `--force` to rebuild).

---

## 10. How to debug a single question yourself

1. **Find it** in the latest run:
   ```bash
   grep -h financebench_id_03029 runs/baseline-*/traces.jsonl | tail -1 | python -m json.tool
   ```
   Look at `status`, `evidence_units`, and the `doc_id`/pages of what was retrieved.
2. **Rung 1, parsing:** open `data/processed/parsed/3M_2018_10K.json`, find
   `"page_index": 59`. Is the evidence text there and readable?
   (`runs/parse_audit/baseline.json` has its coverage.)
3. **Rung 1, chunking:** in `data/processed/chunks/<fp>/3M_2018_10K.json`, find the
   chunks with `page_start ≤ 59 ≤ page_end`. Is the answer inside one chunk, or split?
   (`runs/chunk_audit/baseline.json` has the best single-chunk coverage.)
4. **Rung 2, retrieval:** compare the evidence chunk with the top-ranked chunks in the
   trace. Wrong year/company → filters. Same filing, wrong page → chunk quality,
   keyword search, reranking.

---

## 11. How the tests are built

- **Mirror layout:** `src/rag/x/y.py` → `tests/x/test_y.py`.
- **Fakes, not mocks**, for boundaries: `tests/support/embedders.py` (`HashingEmbedder`:
  hashed bag-of-words, so texts sharing words are similar),
  `tests/support/tokenizers.py` (`WordCounter`, `CharCounter`), ticking clocks, scripted
  retrievers.
- **Real PDFs in memory:** `tests/ingestion/conftest.py` builds tiny valid PDFs (and the
  parser tests encrypt them with pypdf), so parsing is tested for real without fixture
  files or mocking pypdf.
- **Hand-computed expectations** for every metric, including the guide's examples.
- **A property-style test** for the chunker: random documents (seeded) must keep exact
  offsets, the size bound, ordering and every word.
- **`@pytest.mark.slow`** tests use the real tokenizer and the real bge model; they are
  excluded by default (`make test-slow` runs them).
- `pytest` runs in `importlib` mode and `tests/` is on the path, so `from support... import`
  works and same-named test files in different folders don't clash.

Current: 187 fast tests, 2 slow, 95% coverage.

---

## 12. Suggested reading order of the code

1. `src/rag/domain/models.py`: the vocabulary everything else speaks.
2. `src/rag/ingestion/base.py`, `indexing/base.py`, `retrieval/base.py`: the stage contracts.
3. `configs/baseline.yaml` + `src/rag/config/settings.py`: every knob.
4. `src/rag/datasets/financebench.py`: how questions and evidence enter the system.
5. `src/rag/ingestion/pdf_parser.py` → `parse_stage.py`, with `scripts/parse_documents.py`.
6. `src/rag/ingestion/chunkers.py`: the most algorithmic file; read with `tests/ingestion/test_chunkers.py`.
7. `src/rag/indexing/vector_store.py` → `embedders.py` → `index_stage.py`.
8. `src/rag/retrieval/dense.py`: short, on purpose.
9. `src/rag/evaluation/retrieval_metrics.py` with its tests, then `evidence.py`,
   `retrieval_report.py`, `retrieval_runner.py`, and `scripts/evaluate_retrieval.py`.
10. The audits: `parse_audit.py`, `chunk_audit.py`.

Design documents behind each step: `docs/hld/001`…`005`; decisions: `docs/adr/0001`…`0003`.

---

## 13. Known limitations (on purpose, for now)

| Limitation | Why it's acceptable now | Where it gets addressed |
| --- | --- | --- |
| No cleaning; tables flattened | Baseline to measure against | 3.1 |
| Chunks ignore headings/structure, may span pages | Same | 3.2 |
| Chunk sizes measured in tiktoken, 91 chunks truncated by bge | Small (0.35%) | backlog note |
| No metadata filters taken from the question | Shows the version trap clearly | **3.3 (next)** |
| Dense only; numbers and exact terms blur | Same | 3.4 (BM25 + RRF) |
| No reranker | Same | 3.5 |
| Page-level relevance is lenient | Matches FinanceBench labels; chunk audit quantifies it | later, span-level |
| Exact search won't scale to millions | Correct, and the reference for ANN later | 4.2 |
| No generation, no answer metrics | Retrieval must work first | Phase 2 |
| Only 150 questions, none unanswerable | Upstream limitation | our own questions in `evals/` |
