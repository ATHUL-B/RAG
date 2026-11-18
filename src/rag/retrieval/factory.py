"""Config → retriever. Ablations change the config, not the code."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rag.config import BM25Config, FilterField, RetrievalConfig, RetrievalMode
from rag.domain import Chunk
from rag.errors import ConfigError
from rag.indexing.base import Embedder, KeywordIndex, VectorStore
from rag.indexing.keyword_index import BM25Index
from rag.retrieval.base import Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.filtered import FilteredRetriever
from rag.retrieval.filters import CatalogFilterExtractor
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever


def build_bm25_index(config: BM25Config, chunks: Iterable[Chunk]) -> BM25Index:
    index = BM25Index(k1=config.k1, b=config.b)
    index.add(list(chunks))
    return index


def build_retriever(
    config: RetrievalConfig,
    embedder: Embedder,
    store: VectorStore,
    company_aliases: Mapping[str, Iterable[str]],
    keyword_index: KeywordIndex | None = None,
) -> Retriever:
    retriever = _base_retriever(config, embedder, store, keyword_index)
    filters = config.filters
    if filters.enabled:
        extractor = CatalogFilterExtractor(
            company_aliases,
            year_window=filters.year_window,
            extract_company=FilterField.COMPANY in filters.fields,
            extract_year=FilterField.YEAR in filters.fields,
        )
        retriever = FilteredRetriever(
            retriever, extractor, relax_order=[str(f) for f in filters.relax_order]
        )
    return retriever


def _base_retriever(
    config: RetrievalConfig,
    embedder: Embedder,
    store: VectorStore,
    keyword_index: KeywordIndex | None,
) -> Retriever:
    dense = DenseRetriever(embedder, store)
    if config.mode is RetrievalMode.DENSE:
        return dense
    if keyword_index is None:
        raise ConfigError(f"retrieval mode {config.mode!s} needs a keyword index")
    keyword = KeywordRetriever(keyword_index)
    if config.mode is RetrievalMode.BM25:
        return keyword
    fuser = ReciprocalRankFusion(k=config.fusion.rrf_k, weights=config.fusion.weights)
    return HybridRetriever(
        {"dense": dense, "bm25": keyword}, fuser, candidate_depth=config.candidate_depth
    )
