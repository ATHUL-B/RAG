# Architecture

Current state: **parsing and chunking work** (pypdf → `data/processed/parsed/`, recursive
chunker → `data/processed/chunks/<fingerprint>/`, plus parse and chunk audits). No index or
LLM yet (see `backlog.md`).

## Pipelines

```
OFFLINE  raw PDFs ─► Parser ─► Cleaner ─► MetadataEnricher ─► Chunker ─► Embedder ─► VectorStore
                        │          │              │               │                   KeywordIndex
                        ▼          ▼              ▼               ▼
                   data/processed/<stage>/*.jsonl   (inspectable artifact after every stage)

ONLINE   Query ─► QueryRewriter ─► Retriever(s) ─► Fuser ─► Reranker ─► ContextAssembler ─► LLMClient ─► Answer
                                     (filters applied before ranking)                       + trace + StageTimer
```

## Packages

| Package | Responsibility | Key types |
| --- | --- | --- |
| `rag.domain` | Immutable data shared by all stages; deterministic IDs | `Document`, `Page`, `Chunk`, `SourceSpan`, `ScoredChunk`, `Query`, `Answer`, `GoldenItem` |
| `rag.ingestion` | Parse, clean, enrich, chunk | `Parser`, `Cleaner`, `MetadataEnricher`, `Chunker`; `PypdfParser`, `StaticMetadataEnricher`, `ParseStage`, `RecursiveChunker`, `TokenCounter`, `ChunkStage`, `build_parser`, `build_chunker` |
| `rag.indexing` | Embed and index | `Embedder`, `VectorStore`, `KeywordIndex` |
| `rag.retrieval` | Rewrite, retrieve, fuse, rerank | `QueryRewriter`, `Retriever`, `Fuser`, `Reranker` |
| `rag.generation` | Prompt assembly and LLM call | `ContextAssembler`, `LLMClient`, `AnswerGenerator`, `Prompt`, `Completion` |
| `rag.evaluation` | Golden sets, metrics, judges | `retrieval_metrics` (pure functions), `RetrievalEvaluator`, `aggregate`, `MetricsReport`, `evidence` (qrels, relevance units), `ParseAuditor`, `ChunkAuditor`, `Judge`, `Verdict` |
| `rag.observability` | Timing and structured logs | `StageTimer`, `JsonFormatter` |
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
- **Relevance unit = evidence page** (`<doc_id>#p<page>`): a chunk is relevant if it covers an
  evidence page (HLD 003). Lenient by design; the chunk audit measures how much evidence
  actually fits in one chunk.
- **Errors per stage** (`IngestionError`, `RetrievalError`, ...) under `RagError`.
- Dataset choice: see `adr/0001-dataset.md`. Tooling: `adr/0002-tooling.md`.
