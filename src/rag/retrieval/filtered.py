"""Filter-then-retrieve with stepwise relaxation (study guide, Part 8; HLD 006)."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain import Filters, Query
from rag.retrieval.base import FilterExtractor, RetrievalResult, Retriever


class FilteredRetriever(Retriever):
    """Decorates another retriever with filters extracted from the question.

    Explicit ``query.filters`` (e.g. from a user's account or permissions) always win
    over extracted ones and are never relaxed. Extracted filters are relaxed one key at
    a time, in ``relax_order``, while the search returns fewer than ``min_candidates``.
    """

    def __init__(
        self,
        inner: Retriever,
        extractor: FilterExtractor,
        relax_order: Sequence[str] = ("year", "company"),
        min_candidates: int = 1,
    ) -> None:
        if min_candidates < 0:
            raise ValueError("min_candidates must be >= 0")
        self._inner = inner
        self._extractor = extractor
        self._relax_order = tuple(relax_order)
        self._min_candidates = min_candidates

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        extracted = self._extractor.extract(query.text)
        relaxable = [
            key for key in self._relax_order if key in extracted and key not in query.filters
        ]
        filters: Filters = {**extracted, **query.filters}
        relaxed: list[str] = []
        while True:
            result = self._inner.retrieve(query.model_copy(update={"filters": filters}), k)
            if len(result.candidates) >= self._min_candidates or not relaxable:
                return result.model_copy(update={"relaxed": tuple(relaxed)})
            dropped = relaxable.pop(0)
            relaxed.append(dropped)
            filters = {key: value for key, value in filters.items() if key != dropped}
