import pytest

from rag.domain import Chunk, GoldenItem, SourceSpan
from rag.evaluation.chunk_audit import ChunkAuditor


def _chunk(chunk_id: str, page: int, text: str, tokens: int = 10) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=page,
        page_end=page,
        metadata={"token_count": tokens},
    )


CHUNKS = {
    "D": [
        _chunk("c1", 3, "revenue grew ten", tokens=3),
        _chunk("c2", 3, "percent in fiscal", tokens=3),
        _chunk("c3", 4, "capex was 1577 million", tokens=4),
    ]
}


def _item(item_id: str, doc_id: str, page: int, text: str) -> GoldenItem:
    span = SourceSpan(doc_id=doc_id, page_index=page, text=text)
    return GoldenItem(item_id=item_id, question="?", reference_answer="a", evidence=(span,))


def test_reports_best_single_chunk_coverage() -> None:
    report = ChunkAuditor(CHUNKS, min_coverage=0.8).audit(
        [
            _item("split", "D", 3, "revenue grew ten percent"),  # cut across c1 and c2
            _item("whole", "D", 4, "capex was 1577 million"),
            _item("nodoc", "X", 0, "anything"),
            _item("empty", "D", 3, ""),
        ]
    )
    rows = {row.item_id: row for row in report.rows}

    assert rows["split"].chunks_on_page == 2
    assert rows["split"].best_chunk_id == "c1"
    assert rows["split"].best_coverage == pytest.approx(3 / 4)
    assert rows["whole"].best_coverage == 1.0
    assert rows["nodoc"].chunks_on_page == 0
    assert rows["empty"].best_coverage is None

    assert report.fits_in_one_chunk_rate == 0.5
    assert report.mean_best_coverage == pytest.approx((0.75 + 1.0) / 2)
    assert [r.item_id for r in report.missing_doc_rows] == ["nodoc"]
    assert report.chunk_count == 3
    assert sorted(report.token_counts) == [3, 3, 4]
    assert report.token_percentiles()["max"] == 4.0


def test_empty_inputs_give_zero_rates() -> None:
    report = ChunkAuditor({}, min_coverage=0.8).audit([])
    assert report.fits_in_one_chunk_rate == 0.0
    assert report.mean_best_coverage == 0.0
    assert report.token_percentiles() == {}


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="min_coverage"):
        ChunkAuditor({}, min_coverage=-0.1)
