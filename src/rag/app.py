"""Composition root: build the configured retrieval stack from artifacts on disk.

Scripts call this instead of wiring components themselves, so every entry point searches
exactly the same way for a given config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag.config import AppConfig, ChunkingConfig, LLMProvider, RerankerKind, RetrievalMode
from rag.datasets.financebench import COMPANY_ALIASES, FinanceBenchDataset
from rag.domain import Chunk, GoldenItem
from rag.errors import ConfigError
from rag.evaluation.splits import SplitName, load_split, select
from rag.generation.llm_clients import OllamaLLMClient, check_ollama_model
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.retrieval.base import Retriever
from rag.retrieval.factory import build_bm25_index, build_retriever
from rag.retrieval.llm_planner import LLMQueryPlanner, PlanCache, plan_schema
from rag.retrieval.llm_reranker import GradeCache, GradeFormat, LLMPointwiseReranker
from rag.retrieval.parent_child import map_children_to_parents

ModelCheck = Callable[[str, str], None]


@dataclass(frozen=True)
class RetrievalStack:
    retriever: Retriever
    store: ExactVectorStore
    planner: LLMQueryPlanner | None = None
    """Kept so scripts can report planner calls, cache hits and failures."""
    llm_reranker: LLMPointwiseReranker | None = None


def load_retrieval_stack(config: AppConfig) -> RetrievalStack:
    """Load the dense index for this config's chunker + model, then compose the retriever."""
    embedder = build_embedder(config.embedding)
    index_dir = dense_index_dir(
        config, build_chunker(config.chunking, parser=config.parsing.parser).fingerprint
    )
    store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
    children = load_children(config, embedder.model_id, store) if _parent_child(config) else None
    keyword_index = (
        build_bm25_index(config.retrieval.bm25, store.chunks)
        if config.retrieval.mode is not RetrievalMode.DENSE
        else None
    )
    planner = build_planner(config)
    llm_reranker = build_llm_reranker(config)
    retriever = build_retriever(
        config.retrieval,
        embedder,
        store,
        COMPANY_ALIASES,
        keyword_index,
        reranker=llm_reranker,
        planner=planner,
        children=children,
    )
    embedder.embed_query("warm-up")  # load the model before any timed work
    return RetrievalStack(
        retriever=retriever, store=store, planner=planner, llm_reranker=llm_reranker
    )


def _parent_child(config: AppConfig) -> bool:
    return config.retrieval.parent_child.enabled


def child_chunking(config: AppConfig) -> ChunkingConfig:
    """The child chunks: the parents' chunking with the child size and overlap (HLD 023)."""
    pc = config.retrieval.parent_child
    return config.chunking.model_copy(
        update={"chunk_size": pc.child_chunk_size, "chunk_overlap": pc.child_chunk_overlap}
    )


def load_children(
    config: AppConfig, model_id: str, parents: ExactVectorStore
) -> tuple[ExactVectorStore, dict[str, Chunk]]:
    """The child index and the child → parent mapping. The child index must already exist
    (``make data`` with a config whose chunking is :func:`child_chunking`)."""
    fingerprint = build_chunker(child_chunking(config), parser=config.parsing.parser).fingerprint
    index_dir = dense_index_dir(config, fingerprint)
    if not index_dir.exists():
        raise ConfigError(
            f"parent_child needs the child index {index_dir}: build it with a config whose "
            "chunking uses the child size and overlap (e.g. configs/chunks-200-o0.yaml)"
        )
    child_store = ExactVectorStore.load(index_dir, expected_model_id=model_id)
    return child_store, map_children_to_parents(child_store.chunks, parents.chunks)


def build_llm_reranker(config: AppConfig) -> LLMPointwiseReranker | None:
    """The configured local-LLM reranker (HLD 014), or None. No network until first use."""
    rerank = config.retrieval.rerank
    if not (rerank.enabled and rerank.kind is RerankerKind.LLM):
        return None
    llm = OllamaLLMClient(
        model=rerank.llm.model_id,
        base_url=rerank.llm.ollama_url,
        timeout_s=float(rerank.llm.timeout_s),
        max_output_tokens=rerank.llm.max_output_tokens,
        context_tokens=rerank.llm.context_tokens,
    )
    grade_format = GradeFormat(rerank.llm.grade_format)
    cache = (
        GradeCache.for_model(config.paths.rerank_cache_dir, rerank.llm.model_id, grade_format)
        if rerank.llm.cache
        else None
    )
    return LLMPointwiseReranker(
        llm,
        max_passage_chars=rerank.llm.max_passage_chars,
        cache=cache,
        concurrency=rerank.llm.concurrency,
        grade_format=grade_format,
        time_budget_s=rerank.llm.time_budget_s,
    )


def build_planner(config: AppConfig) -> LLMQueryPlanner | None:
    """The configured LLM query planner (local Ollama model, HLD 012), or None."""
    planner = config.retrieval.rewrite.llm_planner
    if not planner.enabled:
        return None
    llm = OllamaLLMClient(
        model=planner.model_id,
        base_url=planner.ollama_url,
        timeout_s=float(planner.timeout_s),
        max_output_tokens=planner.max_output_tokens,
    )
    schema = plan_schema(planner.max_sub_queries, planner.max_chars)
    cache = (
        PlanCache.for_model(config.paths.planner_cache_dir, planner.model_id, schema)
        if planner.cache
        else None
    )
    return LLMQueryPlanner(
        llm, max_sub_queries=planner.max_sub_queries, max_chars=planner.max_chars, cache=cache
    )


def load_golden_items(config: AppConfig, split: SplitName) -> list[GoldenItem]:
    """The golden questions of one split (dev by default in the scripts; see splits.py)."""
    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()
    if split is SplitName.ALL:
        return items
    return select(items, load_split(config.evaluation.split_file), split)


def local_models(config: AppConfig) -> list[tuple[str, str]]:
    """(model, Ollama URL) for every local model this config needs: planner, reranker, generator."""
    needed: list[tuple[str, str]] = []
    planner = config.retrieval.rewrite.llm_planner
    if planner.enabled:
        needed.append((planner.model_id, planner.ollama_url))
    rerank = config.retrieval.rerank
    if rerank.enabled and rerank.kind is RerankerKind.LLM:
        needed.append((rerank.llm.model_id, rerank.llm.ollama_url))
    if config.generation.provider is LLMProvider.OLLAMA:
        needed.append((config.generation.model_id, config.generation.ollama_url))
    return list(dict.fromkeys(needed))


def check_local_models(config: AppConfig, check: ModelCheck = check_ollama_model) -> None:
    """Fail fast (``GenerationError``) if a needed local model is missing.

    A missing model otherwise turns every LLM step into a silent fallback, and the run
    looks like a genuine result.
    """
    for model_id, url in local_models(config):
        check(model_id, url)
