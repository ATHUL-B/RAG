"""Search with the original question *and* its rewrites, then fuse (Part 7; HLD 011)."""

from __future__ import annotations

from rag.domain import Query
from rag.retrieval.base import Fuser, QueryRewriter, RetrievalResult, Retriever

ORIGINAL: str = "original"


class MultiQueryRetriever(Retriever):
    """Runs ``inner`` once per search text and fuses the ranked lists (e.g. with RRF).

    ``keep_original`` follows Part 7's first safety rule: a rewrite can drift, so the
    original question stays one of the searches and anchors the fusion. Filters pass
    through unchanged: they were extracted from the original question by the outer
    ``FilteredRetriever``, so a rewrite cannot drop or invent one. When the rewriter
    changes nothing, this is exactly ``inner`` (no extra search, no extra latency).
    """

    def __init__(
        self,
        inner: Retriever,
        rewriter: QueryRewriter,
        fuser: Fuser,
        keep_original: bool = True,
        candidate_depth: int = 50,
    ) -> None:
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be >= 1")
        self._inner = inner
        self._rewriter = rewriter
        self._fuser = fuser
        self._keep_original = keep_original
        self._candidate_depth = candidate_depth

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        rewrites = [q for q in self._rewriter.rewrite(query) if q.text != query.text]
        if not rewrites:
            return self._inner.retrieve(query, k)
        searches = {f"rewrite_{i}": q for i, q in enumerate(rewrites, start=1)}
        if self._keep_original:
            searches = {ORIGINAL: query, **searches}
        if len(searches) == 1:  # one rewrite, original dropped: no fusion needed
            (only,) = searches.values()
            result = self._inner.retrieve(only, k)
            return result.model_copy(update={"queries": (only.text,)})

        depth = max(k, self._candidate_depth)
        results = {name: self._inner.retrieve(q, depth) for name, q in searches.items()}
        fused = self._fuser.fuse([r.candidates for r in results.values()], k)
        return RetrievalResult(
            candidates=tuple(fused),
            applied_filters=dict(query.filters),
            stage_candidates={name: r.candidates for name, r in results.items()},
            queries=tuple(q.text for q in searches.values()),
        )
