# Architecture

Current state (2026-11-22): **retrieval with metadata filters, page-aware contextual chunks
and a rule-based query planner** by default, plus opt-in LLM stages, grounded generation
with integrity checks, LLM judges, refusal measurement and a latency budget.

| Profile (config) | dev recall@5 | cold time per question | needs |
| --- | --- | --- | --- |
| `planner-rules.yaml` (**default**; HLD 011) | 0.484 (test 0.481) | ~16 ms | nothing |
| `planner-llm.yaml` (LLM query planner; HLD 012) | 0.583 | ~2.7 s | Ollama |
| `planner-llm-rerank-llm-budget13.yaml` (**balanced**: 13 s rerank budget; HLD 020) | 0.688 | 16.3 s p50 / 17.7 s p95 | Ollama |
| `planner-llm-rerank-llm.yaml` (**best quality**: + LLM reranker; HLD 014) | 0.740 (test 0.667) | 28.9 s p50 / 32.4 s p95 | Ollama |
| `planner-llm-rerank-llm-pc.yaml` (best + parent-child, opt-in; HLD 023) | 0.781 (n.s. vs 0.740) | about the same as best | Ollama + 200-token child index |
| any of the above, repeated question (warm caches) | same | ~42 ms | caches |

Generation: llama3.1:8b via Ollama (HLD 015), judged by qwen2.5:7b (HLD 016, provisional),
refusals measured on our own unanswerable set (HLD 017). The generator, not retrieval, is
now the bottleneck: evidence reaches 79% of prompts with the best retrieval, but answers
don't improve (HLD 015, 021, 022).

Measured and **not adopted** (kept opt-in, with configs):
- layout-mode parsing (HLD 021);
- 200 / 800-token chunks and 10% overlap (HLD 022);
- concurrent grading and one-token grades (HLD 019).

Guided tours:
1. [`walkthroughs/phase-1-baseline.md`](walkthroughs/phase-1-baseline.md)
2. [`walkthroughs/phase-3-improvements.md`](walkthroughs/phase-3-improvements.md)
3. [`walkthroughs/phase-3-part-2-planning-and-reranking.md`](walkthroughs/phase-3-part-2-planning-and-reranking.md)
4. [`walkthroughs/phase-2-generation.md`](walkthroughs/phase-2-generation.md) (answers, judges, refusals)
5. [`walkthroughs/phase-4-latency-and-ingestion-experiments.md`](walkthroughs/phase-4-latency-and-ingestion-experiments.md)
   (latency budget, time limits, table parsing, chunk sizes, parent-child)

## Pipelines

```
OFFLINE  raw PDFs ─► Parser ─► Cleaner ─► MetadataEnricher ─► Chunker [+context] ─► Embedder ─► VectorStore
                        │          │              │               │                   KeywordIndex
                        ▼          ▼              ▼               ▼
                   data/processed/<stage>/*.jsonl   (inspectable artifact after every stage)

ONLINE   Query ─► FilteredRetriever (company, year; relaxation) ─► [RerankingRetriever: LLM grades top 20, optional time budget]
              ─► MultiQueryRetriever (original + rule rewrite + [LLM sub-queries], RRF) ─► first stage:
                   Dense | BM25 | Hybrid (RRF) | [ParentChild: dense over child chunks → parent chunks]
         ─► SourcesContextAssembler ([S1]..[S5]) ─► LLMClient (Ollama | Claude | OpenAI-compatible gateway)
         ─► GroundedAnswerGenerator (schema + integrity checks, citations) ─► Answer
         every stage: RetrievalResult (filters, relaxations, stage lists, timings, degraded) → trace
```

## Packages

| Package | Responsibility | Key types |
| --- | --- | --- |
| `rag.domain` | Immutable data shared by all stages; deterministic IDs | `Document`, `Page`, `Chunk`, `SourceSpan`, `ScoredChunk`, `Query`, `Answer`, `GoldenItem` |
| `rag.ingestion` | Parse, clean, enrich, chunk | `Parser`, `Cleaner`, `MetadataEnricher`, `Chunker`; `PypdfParser`, `PypdfLayoutParser` (HLD 021), `StaticMetadataEnricher`, `ParseStage`, `RecursiveChunker` (optionally page-aware), `ContextualChunker`, `HeaderContextualizer`, `TokenCounter`, `ChunkStage`, `build_parser`, `build_chunker` |
| `rag.indexing` | Embed and index | `Embedder`, `VectorStore`, `KeywordIndex`; `SentenceTransformerEmbedder`, `ExactVectorStore`, `BM25Index`, `analyze`, `IndexStage`, `build_embedder` |
| `rag.retrieval` | Extract filters, retrieve, fuse, rerank | `FilterExtractor`, `Retriever` → `RetrievalResult`, `Fuser`, `Reranker`; `DenseRetriever`, `KeywordRetriever`, `HybridRetriever`, `ReciprocalRankFusion`, `CrossEncoderReranker`, `RerankingRetriever`, `CatalogFilterExtractor`, `FilteredRetriever`, `QueryRewriter`, `RuleBasedRewriter`, `LLMQueryPlanner`, `MultiQueryRetriever`, `LLMPointwiseReranker` (grade cache, time budget, concurrency), `ParentChildRetriever` (HLD 023), `build_retriever` |
| `rag.generation` | Prompt assembly and LLM call | `ContextAssembler`, `LLMClient`, `AnswerGenerator`, `Prompt`, `Completion`, `GenerationResult`; `SourcesContextAssembler`, `AnthropicLLMClient`, `OllamaLLMClient`, `OpenAICompatibleLLMClient` (HLD 018), `GroundedAnswerGenerator`, `build_generator` |
| `rag.evaluation` | Golden sets, metrics, judges | `retrieval_metrics` (pure functions, incl. `recall_at_token_budget`, HLD 022), `RetrievalEvaluator`, `aggregate`, `MetricsReport`, `evidence` (qrels, relevance units), `RetrievalEvalRunner`, `QueryTrace`, `GenerationEvalRunner`, `generation_metrics`, `compare_runs` (paired bootstrap), `ParseAuditor`, `ChunkAuditor`, `Judge`, `LLMJudge`, `Verdict`, `cohens_kappa`, splits (dev/test) |
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
- **Generation** (HLD 010, 015, ADR 0005):
  - sources are labelled `[S1]..[Sn]`;
  - the model returns `{answerable, working, answer}` via structured output, and only
    `answer` is scored;
  - citations are parsed and validated against the sources shown;
  - integrity checks reject empty, boolean, markup-leaked and content-filtered replies as
    typed errors, with the raw output kept in the trace.

  Providers sit behind `LLMClient`: Claude via the Anthropic SDK, local Ollama, or any
  OpenAI-compatible gateway (URL and key from the environment only). Paid runs are
  estimated first and blocked above a budget.
- **Dev/test split** (`evals/financebench-split.json`, grouped by company): every setting is
  chosen on dev; test only confirms, and is never used to decide (CLAUDE.md §2).
- **Per-parser artifacts** (HLD 021): parsed documents go to `parsed-<parser>/`, and a
  non-baseline parser joins the chunk fingerprint. Chunk IDs and indexes are only valid
  for the text they were cut from. The baseline keeps its original paths and IDs, so every
  existing cache stays valid.
- **Caches keyed by content and prompt** (planner plans, rerank grades, judge verdicts:
  model + prompt/schema hash in the file name, question/chunk in the key). Changing a prompt
  invalidates its cache automatically; warm runs are ~700× faster than cold (HLD 019).
- **Graceful degradation is visible** (HLD 020, CLAUDE.md §9). A reranker failure or a spent
  time budget falls back to first-stage order *and* is recorded per question in
  `RetrievalResult.degraded` → `QueryTrace.degraded`, never silently.
- **Compare at equal cost** (HLD 022): chunk-size and top-k comparisons use
  `recall_at_token_budget`, not only recall@k; relevance stays page-level, checked against
  the evidence text and the answers before any decision.
- **Relevance unit = evidence page** (`<doc_id>#p<page>`): a chunk is relevant if it covers an
  evidence page (HLD 003). Lenient by design; the chunk audit measures how much evidence
  actually fits in one chunk.
- **Errors per stage** (`IngestionError`, `RetrievalError`, ...) under `RagError`.
- Dense index stored at `data/indexes/dense/<chunker_fp>__<model_slug>/` with a manifest;
  loading with a different model is refused (never mix models).
- Dataset: `adr/0001-dataset.md`. Tooling: `adr/0002-tooling.md`. Embeddings and vector
  store: `adr/0003-embedding-model-and-vector-store.md`.
