from collections.abc import Sequence

import pytest

from rag.domain import Chunk, Filters, Query, ScoredChunk
from rag.retrieval.base import Reranker, RetrievalResult, Retriever
from rag.retrieval.reranking import RerankingRetriever


def _hit(chunk_id: str, rank: int, stage: str = "rrf") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text=chunk_id,
        char_start=0,
        char_end=1,
        page_start=0,
        page_end=0,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank, stage=stage)


class FirstStage(Retriever):
    def __init__(self) -> None:
        self.calls: list[tuple[Filters, int]] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append((dict(query.filters), k))
        hits = tuple(_hit(f"c{i}", i) for i in range(1, k + 1))
        return RetrievalResult(
            candidates=hits,
            applied_filters=dict(query.filters),
            relaxed=("year",),
            stage_candidates={"dense": hits[:2]},
            timings_ms={"dense": 3.0},
            queries=("q", "q clean"),
        )


class ReverseReranker(Reranker):
    """Puts the first-stage *last* candidate first."""

    def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=c.chunk, score=float(r), rank=r, stage="rerank")
            for r, c in enumerate(reversed(candidates), start=1)
        ][:top_n]


class TickClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.1
        return self.now


def test_reranks_a_deeper_first_stage_and_returns_k() -> None:
    inner = FirstStage()
    result = RerankingRetriever(inner, ReverseReranker(), depth=20, clock=TickClock()).retrieve(
        Query(text="q", filters={"company": "3M"}), k=3
    )
    assert inner.calls == [({"company": "3M"}, 20)]
    assert [c.chunk.chunk_id for c in result.candidates] == ["c20", "c19", "c18"]
    assert result.applied_filters == {"company": "3M"}
    assert result.relaxed == ("year",)
    assert result.queries == ("q", "q clean")  # rewritten queries survive into the trace


def test_keeps_first_stage_lists_and_timings_for_the_trace() -> None:
    result = RerankingRetriever(
        FirstStage(), ReverseReranker(), depth=5, clock=TickClock()
    ).retrieve(Query(text="q"), k=2)
    assert [c.chunk.chunk_id for c in result.stage_candidates["first_stage"]] == [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
    ]
    assert "dense" in result.stage_candidates
    assert result.timings_ms == {
        "dense": 3.0,
        "first_stage": pytest.approx(100.0),
        "rerank": pytest.approx(100.0),
    }


def test_depth_is_at_least_k_and_must_be_positive() -> None:
    inner = FirstStage()
    RerankingRetriever(inner, ReverseReranker(), depth=2).retrieve(Query(text="q"), k=10)
    assert inner.calls[0][1] == 10
    with pytest.raises(ValueError, match="depth"):
        RerankingRetriever(inner, ReverseReranker(), depth=0)


class CutShort(ReverseReranker):
    @property
    def last_degradation(self) -> str | None:
        return "rerank_budget: graded 2/5, rest in first-stage order"


def test_reranker_degradation_is_carried_into_the_result_for_the_trace() -> None:
    result = RerankingRetriever(FirstStage(), CutShort(), depth=5).retrieve(Query(text="q"), 3)
    assert result.degraded == ("rerank_budget: graded 2/5, rest in first-stage order",)
    plain = RerankingRetriever(FirstStage(), ReverseReranker(), depth=5).retrieve(
        Query(text="q"), 3
    )
    assert plain.degraded == ()
