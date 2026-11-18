import pytest

from rag.domain import Chunk, Filters, Query, ScoredChunk, matches
from rag.retrieval.base import FilterExtractor, RetrievalResult, Retriever
from rag.retrieval.filtered import FilteredRetriever


def _hit(chunk_id: str, **metadata: str | int) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text="t",
        char_start=0,
        char_end=1,
        page_start=0,
        page_end=0,
        metadata=dict(metadata),
    )
    return ScoredChunk(chunk=chunk, score=1.0, rank=1, stage="dense")


CORPUS = [_hit("3m-2022", company="3M", year=2022), _hit("amd-2018", company="AMD", year=2018)]


class FilteringFake(Retriever):
    """Returns corpus entries matching the query filters; records each call."""

    def __init__(self) -> None:
        self.calls: list[Filters] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append(dict(query.filters))
        hits = tuple(h for h in CORPUS if matches(h.chunk.metadata, query.filters))[:k]
        return RetrievalResult(candidates=hits, applied_filters=dict(query.filters))


class FixedExtractor(FilterExtractor):
    def __init__(self, filters: Filters) -> None:
        self._filters = filters

    def extract(self, text: str) -> Filters:
        return dict(self._filters)


def _retrieve(
    extracted: Filters, explicit: Filters | None = None
) -> tuple[RetrievalResult, FilteringFake]:
    inner = FilteringFake()
    retriever = FilteredRetriever(inner, FixedExtractor(extracted))
    return retriever.retrieve(Query(text="q", filters=explicit or {}), k=5), inner


def test_applies_extracted_filters_when_they_find_results() -> None:
    result, inner = _retrieve({"company": "3M", "year": 2022})
    assert [h.chunk.chunk_id for h in result.candidates] == ["3m-2022"]
    assert result.applied_filters == {"company": "3M", "year": 2022}
    assert result.relaxed == ()
    assert len(inner.calls) == 1


def test_relaxes_year_first_then_company() -> None:
    result, inner = _retrieve({"company": "3M", "year": 2018})
    assert inner.calls == [{"company": "3M", "year": 2018}, {"company": "3M"}]
    assert result.relaxed == ("year",)
    assert [h.chunk.chunk_id for h in result.candidates] == ["3m-2022"]

    result, inner = _retrieve({"company": "NOPE", "year": 1999})
    assert inner.calls[-1] == {}
    assert result.relaxed == ("year", "company")
    assert len(result.candidates) == 2


def test_explicit_filters_win_and_are_never_relaxed() -> None:
    result, inner = _retrieve({"company": "3M", "year": 2018}, explicit={"company": "AMD"})
    assert inner.calls[0] == {"company": "AMD", "year": 2018}
    assert [h.chunk.chunk_id for h in result.candidates] == ["amd-2018"]

    result, inner = _retrieve({"year": 1999}, explicit={"company": "NOPE"})
    assert inner.calls == [{"year": 1999, "company": "NOPE"}, {"company": "NOPE"}]
    assert result.candidates == ()
    assert result.relaxed == ("year",)


def test_no_extracted_filters_means_one_plain_search() -> None:
    result, inner = _retrieve({})
    assert inner.calls == [{}]
    assert len(result.candidates) == 2


def test_rejects_negative_min_candidates() -> None:
    with pytest.raises(ValueError, match="min_candidates"):
        FilteredRetriever(FilteringFake(), FixedExtractor({}), min_candidates=-1)
