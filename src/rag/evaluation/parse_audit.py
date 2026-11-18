"""Parse audit: did the evidence text survive PDF extraction? (HLD 002)

This is the first rung of the debugging ladder (study guide, Part 13): if the
evidence is not in the parsed text, no chunker, retriever or LLM can recover it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from statistics import fmean

from pydantic import BaseModel, ConfigDict

from rag.domain import Document, GoldenItem
from rag.errors import StorageError
from rag.storage import JsonArtifactStore

_TOKEN = re.compile(r"[a-z0-9]+")
_NEIGHBOUR_PAGES = (-1, 1)


def audit_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric runs: robust to the whitespace/punctuation noise of PDFs."""
    return _TOKEN.findall(text.lower())


def token_coverage(evidence: str, page: str) -> float:
    """Share of evidence tokens that appear anywhere on the page (order-insensitive)."""
    evidence_tokens = audit_tokens(evidence)
    if not evidence_tokens:
        raise ValueError("evidence has no tokens")
    page_tokens = set(audit_tokens(page))
    return sum(token in page_tokens for token in evidence_tokens) / len(evidence_tokens)


class AuditStatus(StrEnum):
    PASSED = "passed"
    LOW_COVERAGE = "low_coverage"
    DOC_NOT_PARSED = "doc_not_parsed"
    PAGE_OUT_OF_RANGE = "page_out_of_range"
    NO_EVIDENCE_TEXT = "no_evidence_text"


class EvidenceAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    doc_id: str
    page_index: int
    status: AuditStatus
    coverage: float | None = None
    best_neighbour: tuple[int, float] | None = None
    """(page_index, coverage) of a neighbouring page that matches better: a page-offset hint."""


class ParseAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_coverage: float
    rows: list[EvidenceAudit]

    @property
    def status_counts(self) -> dict[AuditStatus, int]:
        return dict(Counter(row.status for row in self.rows))

    @property
    def measured(self) -> list[EvidenceAudit]:
        return [row for row in self.rows if row.coverage is not None]

    @property
    def pass_rate(self) -> float:
        """Share of measurable evidence whose coverage met the threshold."""
        measured = self.measured
        if not measured:
            return 0.0
        return sum(row.status is AuditStatus.PASSED for row in measured) / len(measured)

    @property
    def mean_coverage(self) -> float:
        measured = self.measured
        return (
            fmean(row.coverage for row in measured if row.coverage is not None) if measured else 0.0
        )

    def worst(self, n: int) -> list[EvidenceAudit]:
        return sorted(self.measured, key=lambda row: row.coverage or 0.0)[:n]


class ParseAuditor:
    def __init__(self, store: JsonArtifactStore[Document], min_coverage: float) -> None:
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be within [0, 1]")
        self._store = store
        self._min_coverage = min_coverage
        self._cache: dict[str, Document | None] = {}

    def audit(self, items: Iterable[GoldenItem]) -> ParseAuditReport:
        rows = [
            self._audit_span(item.item_id, span.doc_id, span.page_index, span.text)
            for item in items
            for span in item.evidence
        ]
        return ParseAuditReport(min_coverage=self._min_coverage, rows=rows)

    def _document(self, doc_id: str) -> Document | None:
        if doc_id not in self._cache:
            try:
                self._cache[doc_id] = self._store.load(doc_id)
            except StorageError:
                self._cache[doc_id] = None
        return self._cache[doc_id]

    def _audit_span(self, item_id: str, doc_id: str, page_index: int, text: str) -> EvidenceAudit:
        row = {"item_id": item_id, "doc_id": doc_id, "page_index": page_index}
        if not audit_tokens(text):
            return EvidenceAudit(**row, status=AuditStatus.NO_EVIDENCE_TEXT)
        document = self._document(doc_id)
        if document is None:
            return EvidenceAudit(**row, status=AuditStatus.DOC_NOT_PARSED)
        pages = {page.page_index: page.text for page in document.pages}
        if page_index not in pages:
            return EvidenceAudit(**row, status=AuditStatus.PAGE_OUT_OF_RANGE)

        coverage = token_coverage(text, pages[page_index])
        neighbours = [
            (page_index + offset, token_coverage(text, pages[page_index + offset]))
            for offset in _NEIGHBOUR_PAGES
            if page_index + offset in pages
        ]
        better = max((n for n in neighbours if n[1] > coverage), key=lambda n: n[1], default=None)
        status = AuditStatus.PASSED if coverage >= self._min_coverage else AuditStatus.LOW_COVERAGE
        return EvidenceAudit(**row, status=status, coverage=coverage, best_neighbour=better)
