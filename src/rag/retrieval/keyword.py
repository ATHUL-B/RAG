"""Keyword (BM25) retrieval (study guide, Part 9)."""

from __future__ import annotations

from rag.domain import Query
from rag.indexing.base import KeywordIndex
from rag.retrieval.base import RetrievalResult, Retriever


class KeywordRetriever(Retriever):
    def __init__(self, index: KeywordIndex) -> None:
        self._index = index

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        hits = self._index.search(query.text, k, query.filters or None)
        return RetrievalResult(candidates=tuple(hits), applied_filters=dict(query.filters))
