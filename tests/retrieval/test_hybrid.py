import pytest

from rag.domain import Chunk, Filters, Query, ScoredChunk
from rag.retrieval.base import RetrievalResult, Retriever
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.retrieval.hybrid import HybridRetriever


def _hit(chunk_id: str, rank: int, stage: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text="t",
        char_start=0,
        char_end=1,
        page_start=0,
        page_end=0,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank, stage=stage)


class RecordingRetriever(Retriever):
    def __init__(self, stage: str, chunk_ids: list[str]) -> None:
        self._stage = stage
        self._chunk_ids = chunk_ids
        self.calls: list[tuple[Filters, int]] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append((dict(query.filters), k))
        hits = tuple(_hit(c, r, self._stage) for r, c in enumerate(self._chunk_ids[:k], 1))
        return RetrievalResult(candidates=hits, applied_filters=dict(query.filters))


def test_fuses_lists_and_keeps_each_stage_for_tracing() -> None:
    dense = RecordingRetriever("dense", ["a", "b", "c"])
    bm25 = RecordingRetriever("bm25", ["c", "d"])
    hybrid = HybridRetriever(
        {"dense": dense, "bm25": bm25}, ReciprocalRankFusion(), candidate_depth=10
    )

    result = hybrid.retrieve(Query(text="q", filters={"year": 2018}), k=2)

    assert [c.chunk.chunk_id for c in result.candidates] == ["c", "a"]  # c is in both lists
    assert {c.stage for c in result.candidates} == {"rrf"}
    assert [c.chunk.chunk_id for c in result.stage_candidates["dense"]] == ["a", "b", "c"]
    assert [c.chunk.chunk_id for c in result.stage_candidates["bm25"]] == ["c", "d"]
    assert result.applied_filters == {"year": 2018}


def test_every_retriever_gets_the_same_filters_and_the_candidate_depth() -> None:
    dense, bm25 = RecordingRetriever("dense", []), RecordingRetriever("bm25", [])
    HybridRetriever({"dense": dense, "bm25": bm25}, ReciprocalRankFusion(), 50).retrieve(
        Query(text="q", filters={"company": "3M"}), k=5
    )
    assert dense.calls == bm25.calls == [({"company": "3M"}, 50)]


def test_depth_is_at_least_k() -> None:
    dense = RecordingRetriever("dense", [])
    HybridRetriever({"dense": dense}, ReciprocalRankFusion(), candidate_depth=3).retrieve(
        Query(text="q"), k=10
    )
    assert dense.calls[0][1] == 10


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever({}, ReciprocalRankFusion())
    with pytest.raises(ValueError, match="candidate_depth"):
        HybridRetriever({"d": RecordingRetriever("d", [])}, ReciprocalRankFusion(), 0)
