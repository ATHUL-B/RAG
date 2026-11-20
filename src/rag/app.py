"""Composition root: build the configured retrieval stack from artifacts on disk.

Scripts call this instead of wiring components themselves, so every entry point searches
exactly the same way for a given config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag.config import AppConfig, LLMProvider, RerankerKind, RetrievalMode
from rag.datasets.financebench import COMPANY_ALIASES, FinanceBenchDataset
from rag.domain import GoldenItem
from rag.evaluation.splits import SplitName, load_split, select
from rag.generation.llm_clients import OllamaLLMClient, check_ollama_model
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.retrieval.base import Retriever
from rag.retrieval.factory import build_bm25_index, build_retriever
from rag.retrieval.llm_planner import LLMQueryPlanner, PlanCache, plan_schema
from rag.retrieval.llm_reranker import GradeCache, LLMPointwiseReranker

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
    index_dir = dense_index_dir(config, build_chunker(config.chunking).fingerprint)
    store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
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
    )
    embedder.embed_query("warm-up")  # load the model before any timed work
    return RetrievalStack(
        retriever=retriever, store=store, planner=planner, llm_reranker=llm_reranker
    )


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
    cache = (
        GradeCache.for_model(config.paths.rerank_cache_dir, rerank.llm.model_id)
        if rerank.llm.cache
        else None
    )
    return LLMPointwiseReranker(llm, max_passage_chars=rerank.llm.max_passage_chars, cache=cache)


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
