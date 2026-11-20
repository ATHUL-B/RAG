from rag.domain import Chunk, Query, ScoredChunk
from rag.retrieval.base import QueryRewriter, RetrievalResult, Retriever
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.retrieval.multi_query import MultiQueryRetriever


def _hit(chunk_id: str, rank: int) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id, doc_id="D", text="t", char_start=0, char_end=1, page_start=0, page_end=0
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank, stage="dense")


class ByText(Retriever):
    """Returns a fixed ranking per query text and records every call."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append((query.text, dict(query.filters), k))
        ids = self._rankings.get(query.text, [])[:k]
        return RetrievalResult(
            candidates=tuple(_hit(c, r) for r, c in enumerate(ids, 1)),
            applied_filters=dict(query.filters),
        )


class Suffix(QueryRewriter):
    def __init__(self, *suffixes: str) -> None:
        self._suffixes = suffixes

    def rewrite(self, query: Query) -> list[Query]:
        return [query.model_copy(update={"text": query.text + s}) for s in self._suffixes]


def _retriever(inner: Retriever, rewriter: QueryRewriter, keep: bool = True) -> MultiQueryRetriever:
    return MultiQueryRetriever(
        inner, rewriter, ReciprocalRankFusion(), keep_original=keep, candidate_depth=10
    )


def test_original_and_rewrite_are_fused_with_shared_filters() -> None:
    inner = ByText({"q": ["a", "b"], "q clean": ["b", "c"]})
    result = _retriever(inner, Suffix(" clean")).retrieve(
        Query(text="q", filters={"year": 2018}), k=3
    )

    assert [c.chunk.chunk_id for c in result.candidates] == ["b", "a", "c"]  # b is in both
    assert inner.calls == [("q", {"year": 2018}, 10), ("q clean", {"year": 2018}, 10)]
    assert result.queries == ("q", "q clean")
    assert set(result.stage_candidates) == {"original", "rewrite_1"}
    assert result.applied_filters == {"year": 2018}


def test_no_rewrite_is_exactly_the_inner_retriever() -> None:
    inner = ByText({"q": ["a"]})
    result = _retriever(inner, Suffix()).retrieve(Query(text="q"), k=5)
    assert inner.calls == [("q", {}, 5)]
    assert result.queries == () and [c.chunk.chunk_id for c in result.candidates] == ["a"]


def test_identical_rewrite_is_ignored() -> None:
    inner = ByText({"q": ["a"]})
    _retriever(inner, Suffix("")).retrieve(Query(text="q"), k=5)
    assert len(inner.calls) == 1


def test_without_the_original_a_single_rewrite_is_searched_alone() -> None:
    inner = ByText({"q clean": ["c"]})
    result = _retriever(inner, Suffix(" clean"), keep=False).retrieve(Query(text="q"), k=5)
    assert inner.calls == [("q clean", {}, 5)]
    assert result.queries == ("q clean",)
    assert [c.chunk.chunk_id for c in result.candidates] == ["c"]


def test_several_rewrites_without_the_original_are_fused() -> None:
    inner = ByText({"q 1": ["a"], "q 2": ["a", "b"]})
    result = _retriever(inner, Suffix(" 1", " 2"), keep=False).retrieve(Query(text="q"), k=5)
    assert result.queries == ("q 1", "q 2")
    assert [c.chunk.chunk_id for c in result.candidates] == ["a", "b"]


def test_rejects_bad_depth() -> None:
    import pytest

    with pytest.raises(ValueError, match="candidate_depth"):
        MultiQueryRetriever(ByText({}), Suffix(), ReciprocalRankFusion(), candidate_depth=0)
