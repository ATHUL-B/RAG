# Architecture

Current state: **retrieval with metadata filters and page-aware contextual chunks** (default
`configs/pages-context.yaml`, recall@5 0.427), plus optional BM25/hybrid and cross-encoder
reranking (Phase 3.2-3.5) (pypdf → recursive chunks → bge-small
embeddings → exact NumPy index → dense top-K), evaluated on FinanceBench with traces and
failure-ladder status. No generation yet (see `backlog.md`). Guided tours:
[`walkthroughs/phase-1-baseline.md`](walkthroughs/phase-1-baseline.md), then
[`walkthroughs/phase-3-improvements.md`](walkthroughs/phase-3-improvements.md).

## Pipelines

```
OFFLINE  raw PDFs ─► Parser ─► Cleaner ─► MetadataEnricher ─► Chunker [+context] ─► Embedder ─► VectorStore
                        │          │              │               │                   KeywordIndex
                        ▼          ▼              ▼               ▼
                   data/processed/<stage>/*.jsonl   (inspectable artifact after every stage)

ONLINE   Query ─► FilterExtractor (company, year) ─► Dense + BM25 (same filters) ─► [Reranker] ─► Fuser ─► Reranker ─► ContextAssembler ─► LLMClient ─► Answer
                                     (filters applied before ranking)                       + trace + StageTimer
```

## Packages

| Package | Responsibility | Key types |
| --- | --- | --- |
| `rag.domain` | Immutable data shared by all stages; deterministic IDs | `Document`, `Page`, `Chunk`, `SourceSpan`, `ScoredChunk`, `Query`, `Answer`, `GoldenItem` |
| `rag.ingestion` | Parse, clean, enrich, chunk | `Parser`, `Cleaner`, `MetadataEnricher`, `Chunker`; `PypdfParser`, `StaticMetadataEnricher`, `ParseStage`, `RecursiveChunker` (optionally page-aware), `ContextualChunker`, `HeaderContextualizer`, `TokenCounter`, `ChunkStage`, `build_parser`, `build_chunker` |
| `rag.indexing` | Embed and index | `Embedder`, `VectorStore`, `KeywordIndex`; `SentenceTransformerEmbedder`, `ExactVectorStore`, `BM25Index`, `analyze`, `IndexStage`, `build_embedder` |
| `rag.retrieval` | Extract filters, retrieve, fuse, rerank | `FilterExtractor`, `Retriever` → `RetrievalResult`, `Fuser`, `Reranker`; `DenseRetriever`, `KeywordRetriever`, `HybridRetriever`, `ReciprocalRankFusion`, `CrossEncoderReranker`, `RerankingRetriever`, `CatalogFilterExtractor`, `FilteredRetriever`, `build_retriever` |
| `rag.generation` | Prompt assembly and LLM call | `ContextAssembler`, `LLMClient`, `AnswerGenerator`, `Prompt`, `Completion`, `GenerationResult`; `SourcesContextAssembler`, `AnthropicLLMClient`, `OllamaLLMClient`, `GroundedAnswerGenerator`, `build_generator` |
| `rag.evaluation` | Golden sets, metrics, judges | `retrieval_metrics` (pure functions), `RetrievalEvaluator`, `aggregate`, `MetricsReport`, `evidence` (qrels, relevance units), `RetrievalEvalRunner`, `QueryTrace`, `GenerationEvalRunner`, `generation_metrics`, `compare_runs` (paired bootstrap), `ParseAuditor`, `ChunkAuditor`, `Judge`, `Verdict` |
| `rag.observability` | Timing, latency percentiles, structured logs | `StageTimer`, `latency_summary`, `JsonFormatter` |
| `rag.config` | Typed YAML config | `AppConfig`, `load_config` |
| `rag.app` | Composition root: load the configured retrieval stack | `load_retrieval_stack` |
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
- **Hybrid**: named inner retrievers run with identical filters; RRF fuses by rank only;
  each inner list is kept in `RetrievalResult.stage_candidates` so traces record which
  retriever found the evidence (`stage_ranks`) and the pool a reranker would see
  (`pool_recall`) (HLD 007). The BM25 index is rebuilt in memory from the indexed chunks.
- **Reranking**: `RerankingRetriever` asks the first stage for `depth` candidates and
  reranks them; `RetrievalResult.timings_ms` carries per-stage latency (first_stage, rerank)
  into traces and run summaries (HLD 008). Composition order: filters → reranker → first stage.
- **Chunk text vs indexed text**: `Chunk.text` is always the exact source span;
  `Chunk.context` (filing + page title block) is prepended only in `Chunk.indexed_text`,
  which is what embedding, BM25 and reranking read (HLD 009).
- **Generation** (HLD 010, ADR 0005): sources are labelled `[S1]..[Sn]`; the model returns
  `{answerable, answer}` via structured output; citations are parsed and validated against
  the sources shown. Providers sit behind `LLMClient` (Claude via the Anthropic SDK, local
  Ollama). Paid runs are estimated first and blocked above a budget.
- **Relevance unit = evidence page** (`<doc_id>#p<page>`): a chunk is relevant if it covers an
  evidence page (HLD 003). Lenient by design; the chunk audit measures how much evidence
  actually fits in one chunk.
- **Errors per stage** (`IngestionError`, `RetrievalError`, ...) under `RagError`.
- Dense index stored at `data/indexes/dense/<chunker_fp>__<model_slug>/` with a manifest;
  loading with a different model is refused (never mix models).
- Dataset: `adr/0001-dataset.md`. Tooling: `adr/0002-tooling.md`. Embeddings and vector
  store: `adr/0003-embedding-model-and-vector-store.md`.
