from pathlib import Path

import pytest

from rag.domain import Document, GoldenItem, Page, SourceSpan
from rag.evaluation.parse_audit import (
    AuditStatus,
    ParseAuditor,
    audit_tokens,
    token_coverage,
)
from rag.storage import JsonArtifactStore


def test_tokens_ignore_case_whitespace_and_punctuation() -> None:
    assert audit_tokens("Capital  Expenditure:\n$1,577") == ["capital", "expenditure", "1", "577"]


def test_coverage_is_share_of_evidence_tokens_on_page() -> None:
    assert token_coverage("net income grew", "Net\nincome was flat") == pytest.approx(2 / 3)


def test_number_formatting_differences_lower_coverage() -> None:
    """Known limitation: '5,363' tokenizes as ['5', '363'], so it does not match '5363'."""
    assert token_coverage("net income 5363", "Net income was 5,363") == pytest.approx(2 / 3)
    assert token_coverage("a b", "nothing here") == 0.0


def test_coverage_needs_evidence_tokens() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        token_coverage("  ...  ", "page")


def _item(item_id: str, *spans: tuple[str, int, str]) -> GoldenItem:
    evidence = tuple(SourceSpan(doc_id=d, page_index=p, text=t) for d, p, t in spans)
    return GoldenItem(item_id=item_id, question="q", reference_answer="a", evidence=evidence)


@pytest.fixture
def auditor(tmp_path: Path) -> ParseAuditor:
    store = JsonArtifactStore(tmp_path, Document)
    pages = ("cover page", "revenue grew 10 percent", "capex was 1577 million")
    store.save(
        "DOC",
        Document(
            doc_id="DOC",
            source_uri="u",
            pages=tuple(Page(page_index=i, text=t) for i, t in enumerate(pages)),
        ),
    )
    return ParseAuditor(store, min_coverage=0.8)


def test_classifies_every_evidence_span(auditor: ParseAuditor) -> None:
    report = auditor.audit(
        [
            _item("pass", ("DOC", 1, "Revenue grew 10 percent")),
            _item("low", ("DOC", 1, "revenue fell sharply")),
            _item("shifted", ("DOC", 1, "capex was 1577 million")),
            _item("nodoc", ("MISSING", 0, "x")),
            _item("range", ("DOC", 9, "x")),
            _item("empty", ("DOC", 0, "")),
        ]
    )
    rows = {row.item_id: row for row in report.rows}

    assert rows["pass"].status is AuditStatus.PASSED
    assert rows["pass"].coverage == 1.0
    assert rows["low"].status is AuditStatus.LOW_COVERAGE
    assert rows["low"].coverage == pytest.approx(1 / 3)
    assert rows["shifted"].best_neighbour == (2, 1.0)
    assert rows["pass"].best_neighbour is None
    assert rows["nodoc"].status is AuditStatus.DOC_NOT_PARSED
    assert rows["range"].status is AuditStatus.PAGE_OUT_OF_RANGE
    assert rows["empty"].status is AuditStatus.NO_EVIDENCE_TEXT


def test_summary_uses_only_measurable_spans(auditor: ParseAuditor) -> None:
    report = auditor.audit(
        [
            _item(
                "q1", ("DOC", 1, "revenue grew 10 percent"), ("DOC", 2, "capex was 1577 dollars")
            ),
            _item("q2", ("MISSING", 0, "x")),
        ]
    )
    assert report.status_counts == {
        AuditStatus.PASSED: 1,
        AuditStatus.LOW_COVERAGE: 1,
        AuditStatus.DOC_NOT_PARSED: 1,
    }
    assert report.pass_rate == 0.5
    assert report.mean_coverage == pytest.approx((1.0 + 0.75) / 2)
    assert [row.coverage for row in report.worst(1)] == [0.75]


def test_empty_report_has_zero_rates(auditor: ParseAuditor) -> None:
    report = auditor.audit([])
    assert report.pass_rate == 0.0
    assert report.mean_coverage == 0.0


def test_rejects_invalid_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_coverage"):
        ParseAuditor(JsonArtifactStore(tmp_path, Document), min_coverage=1.5)
