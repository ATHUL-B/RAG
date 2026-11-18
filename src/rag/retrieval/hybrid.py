"""Hybrid retrieval: several retrievers, one fused ranking (study guide, Part 9; HLD 007)."""

from __future__ import annotations

from collections.abc import Mapping

from rag.domain import Query
from rag.retrieval.base import Fuser, RetrievalResult, Retriever


class HybridRetriever(Retriever):
    """Runs every named inner retriever with the same query *and filters*, then fuses.

    Each inner retriever returns ``candidate_depth`` candidates (the guide suggests
    50-100), deeper than the final ``k``, so a chunk ranked 20th by one retriever and
    3rd by another can still win the fusion. Each inner list is kept in
    ``stage_candidates`` under its name, so traces show which retriever found what.
    """

    def __init__(
        self, retrievers: Mapping[str, Retriever], fuser: Fuser, candidate_depth: int = 50
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be >= 1")
        self._retrievers = dict(retrievers)
        self._fuser = fuser
        self._candidate_depth = candidate_depth

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        depth = max(k, self._candidate_depth)
        results = {name: r.retrieve(query, depth) for name, r in self._retrievers.items()}
        fused = self._fuser.fuse([result.candidates for result in results.values()], k)
        return RetrievalResult(
            candidates=tuple(fused),
            applied_filters=dict(query.filters),
            stage_candidates={name: result.candidates for name, result in results.items()},
        )
