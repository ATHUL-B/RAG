"""Search with the original question *and* its rewrites, then fuse (Part 7; HLD 011)."""

from __future__ import annotations

import time

from rag.domain import Query
from rag.observability import StageTimer
from rag.observability.timing import Clock
from rag.retrieval.base import Fuser, QueryRewriter, RetrievalResult, Retriever

ORIGINAL: str = "original"
REWRITE_STAGE: str = "rewrite"


class MultiQueryRetriever(Retriever):
    """Runs ``inner`` once per search text and fuses the ranked lists (e.g. with RRF).

    ``keep_original`` follows Part 7's first safety rule: a rewrite can drift, so the
    original question stays one of the searches and anchors the fusion. Filters pass
    through unchanged: they were extracted from the original question by the outer
    ``FilteredRetriever``, so a rewrite cannot drop or invent one. When the rewriter
    changes nothing, this is exactly ``inner`` (no extra search, no extra latency).

    Each list is labelled with its search name (``original``, ``rewrite_1``, …) before
    fusion, so a weighted fuser can trust the original question more than the rewrites:
    with many rewrites, agreeing sub-queries can otherwise outvote the original's correct
    page (HLD 012, 013).
    """

    def __init__(
        self,
        inner: Retriever,
        rewriter: QueryRewriter,
        fuser: Fuser,
        keep_original: bool = True,
        candidate_depth: int = 50,
        clock: Clock = time.perf_counter,
    ) -> None:
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be >= 1")
        self._inner = inner
        self._rewriter = rewriter
        self._fuser = fuser
        self._keep_original = keep_original
        self._candidate_depth = candidate_depth
        self._clock = clock

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        timer = StageTimer(self._clock)
        with timer.stage(REWRITE_STAGE):  # an LLM planner makes this the slow part
            rewrites = [q for q in self._rewriter.rewrite(query) if q.text != query.text]
        timings = dict(timer.timings_ms)
        if not rewrites:
            result = self._inner.retrieve(query, k)
            return result.model_copy(update={"timings_ms": {**result.timings_ms, **timings}})
        searches = {f"rewrite_{i}": q for i, q in enumerate(rewrites, start=1)}
        if self._keep_original:
            searches = {ORIGINAL: query, **searches}
        if len(searches) == 1:  # one rewrite, original dropped: no fusion needed
            (only,) = searches.values()
            result = self._inner.retrieve(only, k)
            return result.model_copy(
                update={"queries": (only.text,), "timings_ms": {**result.timings_ms, **timings}}
            )

        depth = max(k, self._candidate_depth)
        results = {name: self._inner.retrieve(q, depth) for name, q in searches.items()}
        # Label each list by its search so the fuser can weight it (e.g. "original": 2.0).
        labelled = [
            [c.model_copy(update={"stage": name}) for c in r.candidates]
            for name, r in results.items()
        ]
        fused = self._fuser.fuse(labelled, k)
        return RetrievalResult(
            candidates=tuple(fused),
            applied_filters=dict(query.filters),
            stage_candidates={name: r.candidates for name, r in results.items()},
            queries=tuple(q.text for q in searches.values()),
            timings_ms=timings,
        )
