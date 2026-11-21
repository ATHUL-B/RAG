import pytest

from rag.domain import Chunk, Query, ScoredChunk
from rag.retrieval.base import RetrievalResult, Retriever
from rag.retrieval.parent_child import (
    CHILD_STAGE,
    STAGE,
    ParentChildRetriever,
    map_children_to_parents,
)


def _chunk(chunk_id: str, start: int, end: int, doc_id: str = "D", page: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        page_start=page,
        page_end=page,
    )


PARENTS = [_chunk("P1", 0, 400), _chunk("P2", 400, 800), _chunk("Q1", 0, 300, doc_id="E")]


def test_each_child_maps_to_the_parent_it_overlaps_most() -> None:
    children = [
        _chunk("c1", 0, 200),  # inside P1
        _chunk("c2", 350, 550),  # 50 chars in P1, 150 in P2 -> P2
        _chunk("c3", 600, 800),  # inside P2
        _chunk("e1", 100, 200, doc_id="E"),  # other document: Q1, never P1
        _chunk("orphan", 0, 50, doc_id="Z"),  # no parent in its document
    ]
    mapping = map_children_to_parents(children, PARENTS)
    assert {k: v.chunk_id for k, v in mapping.items()} == {
        "c1": "P1",
        "c2": "P2",
        "c3": "P2",
        "e1": "Q1",
    }


class FixedChildren(Retriever):
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self.asked: list[int] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.asked.append(k)
        hits = tuple(
            ScoredChunk(chunk=_chunk(i, 0, 1), score=1.0 - r / 10, rank=r, stage="dense")
            for r, i in enumerate(self._ids[:k], start=1)
        )
        return RetrievalResult(
            candidates=hits, applied_filters={"company": "3M"}, stage_candidates={"dense": hits}
        )


def test_returns_distinct_parents_in_best_child_order_with_the_best_score() -> None:
    parent_of = {"c1": PARENTS[1], "c2": PARENTS[0], "c3": PARENTS[1], "c4": PARENTS[2]}
    children = FixedChildren(["c1", "c2", "c3", "unmapped", "c4"])
    result = ParentChildRetriever(children, parent_of, fanout=3).retrieve(Query(text="q"), 2)
    assert children.asked == [6]  # k * fanout children
    assert [c.chunk.chunk_id for c in result.candidates] == ["P2", "P1"]
    assert [c.rank for c in result.candidates] == [1, 2]
    assert {c.stage for c in result.candidates} == {STAGE}
    assert result.candidates[0].score == pytest.approx(0.9)  # c1's score, not c3's
    assert result.applied_filters == {"company": "3M"}  # filters pass through
    assert [c.chunk.chunk_id for c in result.stage_candidates[CHILD_STAGE]][:3] == [
        "c1",
        "c2",
        "c3",
    ]


def test_fewer_parents_than_k_when_children_run_out() -> None:
    parent_of = {"c1": PARENTS[0], "c2": PARENTS[0]}
    result = ParentChildRetriever(FixedChildren(["c1", "c2"]), parent_of).retrieve(
        Query(text="q"), 5
    )
    assert [c.chunk.chunk_id for c in result.candidates] == ["P1"]


def test_rejects_bad_fanout() -> None:
    with pytest.raises(ValueError, match="fanout"):
        ParentChildRetriever(FixedChildren([]), {}, fanout=0)
