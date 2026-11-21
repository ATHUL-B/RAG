"""Two-stage retrieval: cheap wide first stage, precise reranker on top (Part 10; HLD 008)."""

from __future__ import annotations

import time

from rag.domain import Query
from rag.observability import StageTimer
from rag.observability.timing import Clock
from rag.retrieval.base import Reranker, RetrievalResult, Retriever

FIRST_STAGE = "first_stage"
RERANK_STAGE = "rerank"


class RerankingRetriever(Retriever):
    """Fetch ``depth`` candidates from ``inner``, rerank them, return the best ``k``.

    The reranker can only reorder what it is given, so ``depth`` bounds the achievable
    recall (pool recall) and drives the latency (one cross-encoder pass per candidate).
    """

    def __init__(
        self,
        inner: Retriever,
        reranker: Reranker,
        depth: int = 50,
        clock: Clock = time.perf_counter,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self._inner = inner
        self._reranker = reranker
        self._depth = depth
        self._clock = clock

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        timer = StageTimer(self._clock)
        with timer.stage(FIRST_STAGE):
            first = self._inner.retrieve(query, max(k, self._depth))
        with timer.stage(RERANK_STAGE):
            reranked = self._reranker.rerank(query, first.candidates, top_n=k)
        note = self._reranker.last_degradation
        return RetrievalResult(
            candidates=tuple(reranked),
            applied_filters=dict(first.applied_filters),
            relaxed=first.relaxed,
            stage_candidates={**first.stage_candidates, FIRST_STAGE: first.candidates},
            timings_ms={**first.timings_ms, **timer.timings_ms},
            queries=first.queries,
            degraded=(*first.degraded, note) if note else first.degraded,
        )
