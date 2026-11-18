import math

import pytest

from rag.domain import Chunk
from rag.errors import IndexingError
from rag.indexing.keyword_index import BM25Index, bm25_idf


def _chunk(chunk_id: str, text: str, **metadata: str | int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text=text,
        char_start=0,
        char_end=max(len(text), 1),
        page_start=0,
        page_end=0,
        metadata=dict(metadata),
    )


def _ids(index: BM25Index, query: str, k: int = 10, **filters: str | int) -> list[str]:
    return [h.chunk.chunk_id for h in index.search(query, k, dict(filters) or None)]


class TestGuideNumbers:
    """Study guide Part 9, 'Worked numbers' (N = 1,000,000 chunks)."""

    def test_idf_of_common_and_rare_terms(self) -> None:
        assert bm25_idf(1_000_000, 40_000) == pytest.approx(math.log(25.0), abs=1e-4)  # 3.22
        assert bm25_idf(1_000_000, 300) == pytest.approx(8.11, abs=0.01)

    def test_term_frequency_saturates(self) -> None:
        # Two chunks of equal (= average) length; "waiting" appears twice in one.
        index = BM25Index(k1=1.2, b=0.75)
        index.add(
            [_chunk("a", "waiting waiting alpha beta"), _chunk("b", "gamma delta epsilon zeta")]
        )
        idf = bm25_idf(2, 1)
        assert index.scores("waiting")[0] == pytest.approx(idf * 2 * 2.2 / (2 + 1.2))  # x 1.375
        assert index.scores("waiting")[1] == 0.0


class TestRanking:
    @pytest.fixture
    def index(self) -> BM25Index:
        index = BM25Index()
        index.add(
            [
                _chunk("capex18", "purchases of property plant and equipment 1,577", year=2018),
                _chunk("capex17", "purchases of property plant and equipment 1,373", year=2017),
                _chunk("div", "dividends paid to shareholders", year=2018),
                _chunk("long", "property " + "filler " * 50, year=2018),
            ]
        )
        return index

    def test_rare_exact_number_decides_the_ranking(self, index: BM25Index) -> None:
        assert _ids(index, "property plant equipment 1577")[0] == "capex18"
        assert _ids(index, "property plant equipment 1373")[0] == "capex17"

    def test_longer_chunks_are_normalized_down(self, index: BM25Index) -> None:
        ranked = _ids(index, "property")
        assert ranked.index("long") > ranked.index("capex18")

    def test_only_chunks_sharing_a_term_are_returned(self, index: BM25Index) -> None:
        assert _ids(index, "dividends") == ["div"]
        assert _ids(index, "unrelated words") == []
        assert _ids(index, "the of and") == []  # stop words only

    def test_filters_apply_before_ranking(self, index: BM25Index) -> None:
        assert _ids(index, "property plant equipment 1577", year=2017) == ["capex17"]

    def test_k_limits_and_ranks_are_sequential(self, index: BM25Index) -> None:
        hits = index.search("property", 2)
        assert [h.rank for h in hits] == [1, 2]
        assert all(h.stage == "bm25" for h in hits)

    def test_repeated_query_terms_count_once(self, index: BM25Index) -> None:
        assert index.scores("dividends dividends")[2] == index.scores("dividends")[2]


class TestBuilding:
    def test_incremental_adds_equal_one_batch(self) -> None:
        chunks = [
            _chunk("a", "net income rose"),
            _chunk("b", "net sales fell"),
            _chunk("c", "income"),
        ]
        one = BM25Index()
        one.add(chunks)
        two = BM25Index()
        two.add(chunks[:1])
        two.add(chunks[1:])
        assert list(one.scores("net income")) == pytest.approx(list(two.scores("net income")))
        assert len(two) == 3
        assert two.vocabulary_size == 5

    def test_rejects_duplicates_and_bad_parameters(self) -> None:
        index = BM25Index()
        index.add([_chunk("a", "x")])
        with pytest.raises(IndexingError, match="already indexed"):
            index.add([_chunk("a", "y")])
        with pytest.raises(ValueError, match="k1"):
            BM25Index(k1=-1)
        with pytest.raises(ValueError, match="b must"):
            BM25Index(b=1.5)
        with pytest.raises(ValueError, match="k must"):
            index.search("x", 0)

    def test_empty_index_returns_nothing(self) -> None:
        assert BM25Index().search("anything", 5) == []


def test_context_header_is_searchable() -> None:
    chunk = _chunk("c", "Acquisitions net of cash acquired").model_copy(
        update={"context": "Consolidated Statement of Cash Flows"}
    )
    index = BM25Index()
    index.add([chunk, _chunk("other", "unrelated words")])
    assert _ids(index, "statement cash flows") == ["c"]
