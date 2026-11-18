import pytest
from pydantic import ValidationError

from rag.domain import (
    PAGE_SEPARATOR,
    Chunk,
    ChunkSet,
    Document,
    GoldenItem,
    Metadata,
    Page,
    Query,
    SourceSpan,
    matches,
)


def _doc(*texts: str, start: int = 0) -> Document:
    pages = tuple(Page(page_index=start + i, text=t) for i, t in enumerate(texts))
    return Document(doc_id="d1", source_uri="file://d1.pdf", pages=pages)


class TestDocument:
    def test_text_joins_pages_with_separator(self) -> None:
        assert _doc("abc", "de").text == f"abc{PAGE_SEPARATOR}de"

    def test_page_spans_locate_each_page_in_text(self) -> None:
        doc = _doc("abc", "de", start=10)
        assert doc.page_spans() == [(10, 0, 3), (11, 5, 7)]
        assert [doc.text[s:e] for _, s, e in doc.page_spans()] == ["abc", "de"]

    def test_page_index_at_maps_offsets_to_pages(self) -> None:
        doc = _doc("abc", "de", start=10)
        second_page_start = len("abc") + len(PAGE_SEPARATOR)
        assert doc.page_index_at(0) == 10
        assert doc.page_index_at(2) == 10
        assert doc.page_index_at(3) == 10  # separator belongs to preceding page
        assert doc.page_index_at(second_page_start) == 11
        assert doc.page_index_at(len(doc.text) - 1) == 11

    @pytest.mark.parametrize("offset", [-1, 100])
    def test_page_index_at_rejects_out_of_range(self, offset: int) -> None:
        with pytest.raises(ValueError, match="outside document"):
            _doc("abc").page_index_at(offset)

    def test_page_index_at_rejects_empty_document(self) -> None:
        with pytest.raises(ValueError, match="no pages"):
            _doc().page_index_at(0)

    def test_pages_must_be_ascending_and_unique(self) -> None:
        pages = (Page(page_index=1, text="b"), Page(page_index=0, text="a"))
        with pytest.raises(ValidationError, match="unique and ascending"):
            Document(doc_id="d", source_uri="u", pages=pages)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _doc("a").doc_id = "other"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            Document(doc_id="d", source_uri="u", pages=(), color="red")  # type: ignore[call-arg]


class TestChunk:
    def _chunk(self, **overrides: int) -> Chunk:
        fields = {"char_start": 0, "char_end": 5, "page_start": 0, "page_end": 0} | overrides
        return Chunk(chunk_id="c", doc_id="d", text="hello", **fields)

    def test_valid_chunk(self) -> None:
        assert self._chunk().char_end == 5

    def test_rejects_empty_span(self) -> None:
        with pytest.raises(ValidationError, match="char_end"):
            self._chunk(char_end=0)

    def test_rejects_inverted_pages(self) -> None:
        with pytest.raises(ValidationError, match="page_end"):
            self._chunk(page_start=2, page_end=1)


def test_golden_item_without_evidence_is_unanswerable() -> None:
    unanswerable = GoldenItem(item_id="q", question="?", reference_answer="", evidence=())
    answerable = unanswerable.model_copy(
        update={"evidence": (SourceSpan(doc_id="d", page_index=0),)}
    )
    assert not unanswerable.answerable
    assert answerable.answerable


def test_chunk_set_rejects_chunks_from_other_documents() -> None:
    chunk = Chunk(
        chunk_id="c", doc_id="OTHER", text="t", char_start=0, char_end=1, page_start=0, page_end=0
    )
    with pytest.raises(ValidationError, match="other documents"):
        ChunkSet(doc_id="d1", chunker_fingerprint="fp", chunks=(chunk,))


META: Metadata = {"company": "3M", "year": 2018}


class TestMatches:
    def test_scalar_means_equals(self) -> None:
        assert matches(META, {"company": "3M"})
        assert not matches(META, {"company": "AMD"})

    def test_tuple_means_any_of(self) -> None:
        assert matches(META, {"year": (2017, 2018)})
        assert not matches(META, {"year": (2019, 2020)})

    def test_all_filters_must_hold_and_missing_keys_fail(self) -> None:
        assert matches(META, {})
        assert not matches(META, {"company": "3M", "year": 2019})
        assert not matches(META, {"sector": "Industrials"})


def test_query_filters_round_trip_through_json() -> None:
    query = Query(text="q", filters={"company": "3M", "year": (2017, 2018)})
    assert Query.model_validate_json(query.model_dump_json()) == query
