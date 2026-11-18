from rag.domain import Chunk, GoldenItem, SourceSpan
from rag.evaluation.evidence import covers, evidence_unit, qrels_for, relevance_ids
from rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k


def _chunk(chunk_id: str, doc_id: str, pages: tuple[int, int]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="t",
        char_start=0,
        char_end=1,
        page_start=pages[0],
        page_end=pages[1],
    )


ITEM = GoldenItem(
    item_id="q",
    question="?",
    reference_answer="a",
    evidence=(
        SourceSpan(doc_id="D", page_index=5),
        SourceSpan(doc_id="D", page_index=9),
        SourceSpan(doc_id="D", page_index=9),  # duplicate page: one unit
    ),
)


def test_qrels_has_one_unit_per_distinct_evidence_page() -> None:
    assert qrels_for(ITEM) == {"D#p5": 1, "D#p9": 1}
    assert evidence_unit("D", 5) == "D#p5"


def test_covers_checks_document_and_page_range() -> None:
    span = SourceSpan(doc_id="D", page_index=5)
    assert covers(_chunk("c", "D", (4, 5)), span)
    assert not covers(_chunk("c", "D", (6, 7)), span)
    assert not covers(_chunk("c", "OTHER", (5, 5)), span)


def test_relevance_ids_map_chunks_to_units_or_keep_their_id() -> None:
    ranked = [
        _chunk("c1", "D", (9, 9)),
        _chunk("c2", "D", (1, 1)),
        _chunk("c3", "OTHER", (5, 5)),  # right page, wrong document (a version trap)
        _chunk("c4", "D", (8, 9)),
    ]
    assert relevance_ids(ranked, ITEM) == ["D#p9", "c2", "c3", "D#p9"]


def test_end_to_end_with_metrics() -> None:
    ranked = [_chunk("c1", "D", (9, 9)), _chunk("c2", "D", (9, 9)), _chunk("c3", "D", (1, 1))]
    ids = relevance_ids(ranked, ITEM)
    assert recall_at_k(ids, qrels_for(ITEM), 3) == 0.5  # page 5 never retrieved
    assert precision_at_k(ids, qrels_for(ITEM), 3) == 2 / 3
