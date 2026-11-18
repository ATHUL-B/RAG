# Architecture

Current state: **end-to-end retrieval baseline + metadata filters** (Phase 3.3) (pypdf → recursive chunks → bge-small
embeddings → exact NumPy index → dense top-K), evaluated on FinanceBench with traces and
failure-ladder status. No generation yet (see `backlog.md`). Guided tour:
[`walkthroughs/phase-1-baseline.md`](walkthroughs/phase-1-baseline.md).

## Pipelines

```
OFFLINE  raw PDFs ─► Parser ─► Cleaner ─► MetadataEnricher ─► Chunker ─► Embedder ─► VectorStore
                        │          │              │               │                   KeywordIndex
                        ▼          ▼              ▼               ▼
                   data/processed/<stage>/*.jsonl   (inspectable artifact after every stage)

ONLINE   Query ─► FilterExtractor (company, year) ─► Retriever(s) ─► Fuser ─► Reranker ─► ContextAssembler ─► LLMClient ─► Answer
                                     (filters applied before ranking)                       + trace + StageTimer
```

## Packages

| Package | Responsibility | Key types |
| --- | --- | --- |
| `rag.domain` | Immutable data shared by all stages; deterministic IDs | `Document`, `Page`, `Chunk`, `SourceSpan`, `ScoredChunk`, `Query`, `Answer`, `GoldenItem` |
| `rag.ingestion` | Parse, clean, enrich, chunk | `Parser`, `Cleaner`, `MetadataEnricher`, `Chunker`; `PypdfParser`, `StaticMetadataEnricher`, `ParseStage`, `RecursiveChunker`, `TokenCounter`, `ChunkStage`, `build_parser`, `build_chunker` |
| `rag.indexing` | Embed and index | `Embedder`, `VectorStore`, `KeywordIndex`; `SentenceTransformerEmbedder`, `ExactVectorStore`, `IndexStage`, `build_embedder` |
| `rag.retrieval` | Extract filters, retrieve, fuse, rerank | `FilterExtractor`, `Retriever` → `RetrievalResult`, `Fuser`, `Reranker`; `DenseRetriever`, `CatalogFilterExtractor`, `FilteredRetriever`, `build_retriever` |
| `rag.generation` | Prompt assembly and LLM call | `ContextAssembler`, `LLMClient`, `AnswerGenerator`, `Prompt`, `Completion` |
| `rag.evaluation` | Golden sets, metrics, judges | `retrieval_metrics` (pure functions), `RetrievalEvaluator`, `aggregate`, `MetricsReport`, `evidence` (qrels, relevance units), `RetrievalEvalRunner`, `QueryTrace`, `ParseAuditor`, `ChunkAuditor`, `Judge`, `Verdict` |
| `rag.observability` | Timing, latency percentiles, structured logs | `StageTimer`, `latency_summary`, `JsonFormatter` |
| `rag.config` | Typed YAML config | `AppConfig`, `load_config` |
| `rag.storage` | Atomic writes, safe keys, JSON artifacts | `JsonArtifactStore`, `atomic_write_bytes` |
| `rag.datasets` | Public dataset adapters | `FinanceBenchDataset`, `FinanceBenchDownloader`, `Fetcher` |

## Key decisions

- **Interfaces per stage (ABCs) + config selection.** An ablation swaps exactly one
  implementation. Collaborators are injected, so unit tests use fakes.
- **Immutable pydantic domain models.** Stages return new objects; artifacts on disk
  are exactly what the next stage saw. `extra="forbid"` catches schema drift early.
- **Evidence labelled at source level** (`SourceSpan`: doc + zero-based page), mapped
  to chunks at eval time, so golden labels survive re-chunking.
- **Deterministic chunk IDs** = hash(doc_id, char span, chunker config fingerprint).
- **Filters**: `Filters` values are scalars (equals) or tuples (any of); `domain.matches` is
  the single definition used by the store's pre-filter and by the evaluator. Retrievers
  return `RetrievalResult` (candidates + applied filters + relaxations) so traces explain
  every search (HLD 006).
- **Relevance unit = evidence page** (`<doc_id>#p<page>`): a chunk is relevant if it covers an
  evidence page (HLD 003). Lenient by design; the chunk audit measures how much evidence
  actually fits in one chunk.
- **Errors per stage** (`IngestionError`, `RetrievalError`, ...) under `RagError`.
- Dense index stored at `data/indexes/dense/<chunker_fp>__<model_slug>/` with a manifest;
  loading with a different model is refused (never mix models).
- Dataset: `adr/0001-dataset.md`. Tooling: `adr/0002-tooling.md`. Embeddings and vector
  store: `adr/0003-embedding-model-and-vector-store.md`.
