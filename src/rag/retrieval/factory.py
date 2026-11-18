"""Config → retriever. Ablations change the config, not the code."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rag.config import FilterField, RetrievalConfig
from rag.indexing.base import Embedder, VectorStore
from rag.retrieval.base import Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.filtered import FilteredRetriever
from rag.retrieval.filters import CatalogFilterExtractor


def build_retriever(
    config: RetrievalConfig,
    embedder: Embedder,
    store: VectorStore,
    company_aliases: Mapping[str, Iterable[str]],
) -> Retriever:
    retriever: Retriever = DenseRetriever(embedder, store)
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
