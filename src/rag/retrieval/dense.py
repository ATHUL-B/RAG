"""Dense (embedding) retrieval: the naive top-K baseline of the study guide, Part 6."""

from __future__ import annotations

from rag.domain import Query
from rag.indexing.base import Embedder, VectorStore
from rag.retrieval.base import RetrievalResult, Retriever


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        vector = self._embedder.embed_query(query.text)
        hits = self._store.search(vector, k, query.filters or None)
        return RetrievalResult(candidates=tuple(hits), applied_filters=dict(query.filters))
