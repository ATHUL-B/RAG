import pytest

from rag.domain import Chunk, GoldenItem, SourceSpan
from rag.evaluation.generation_metrics import (
    cites_evidence,
    numbers_in,
    numeric_match,
    reference_number,
)


@pytest.mark.parametrize(
    ("reference", "value"),
    [
        ("$1577.00", 1577.0),
        ("8.70", 8.7),
        ("1.9%", 1.9),
        ("$1,577", 1577.0),
        ("-3.2%", -3.2),
        ("(4.5)", -4.5),
        ("No, it did not.", None),
        ("The ratio was 0.96", None),
    ],
)
def test_reference_number_only_for_plain_numbers(reference: str, value: float | None) -> None:
    assert reference_number(reference) == value


def test_numbers_in_text() -> None:
    text = "Capex was $1,577 million in FY2018 (vs 1,373), margin 24.26%."
    # "FY2018" is an identifier glued to letters, not a value: it must not count as a number
    assert numbers_in(text) == [1577.0, 1373.0, 24.26]


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("FY2018 capex was $1,577 million [S1].", 1577.0, True),
        ("The ratio is 8.7 [S2].", 8.70, True),
        ("About 1,570 million", 1577.0, True),  # within 1%
        ("Capex was 1,373 million", 1577.0, False),
        ("Declined by 3.2%", -3.2, True),  # sign-insensitive
        ("No numbers here", 12.0, False),
    ],
)
def test_numeric_match(answer: str, reference: float, expected: bool) -> None:
    assert numeric_match(answer, reference) is expected


def test_cites_evidence_is_page_level() -> None:
    item = GoldenItem(
        item_id="q",
        question="?",
        reference_answer="a",
        evidence=(SourceSpan(doc_id="D", page_index=5),),
    )

    def chunk(page: int) -> Chunk:
        return Chunk(
            chunk_id=f"c{page}",
            doc_id="D",
            text="t",
            char_start=0,
            char_end=1,
            page_start=page,
            page_end=page,
        )

    assert cites_evidence([chunk(2), chunk(5)], item)
    assert not cites_evidence([chunk(2)], item)
    assert not cites_evidence([], item)
