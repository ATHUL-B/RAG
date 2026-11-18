"""Chunk audit: does each piece of evidence fit inside one chunk? (HLD 003)

If the answer is cut across chunks (study guide, Part 4), retrieval can only ever
return part of it. This audit runs *before* any retrieval, so a low score here is an
ingestion problem, not a retrieval one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, quantiles

from pydantic import BaseModel, ConfigDict

from rag.domain import Chunk, GoldenItem
from rag.evaluation.evidence import covers
from rag.evaluation.parse_audit import audit_tokens, token_coverage


class EvidenceChunkAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    doc_id: str
    page_index: int
    chunks_on_page: int
    best_chunk_id: str | None = None
    best_coverage: float | None = None
    """Highest share of evidence tokens found in a single chunk covering the page."""


class ChunkAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_coverage: float
    rows: list[EvidenceChunkAudit]
    chunk_count: int
    token_counts: list[int]

    @property
    def measured(self) -> list[EvidenceChunkAudit]:
        return [row for row in self.rows if row.best_coverage is not None]

    @property
    def fits_in_one_chunk_rate(self) -> float:
        measured = self.measured
        if not measured:
            return 0.0
        hits = sum((row.best_coverage or 0.0) >= self.min_coverage for row in measured)
        return hits / len(measured)

    @property
    def mean_best_coverage(self) -> float:
        measured = self.measured
        return fmean(row.best_coverage or 0.0 for row in measured) if measured else 0.0

    @property
    def missing_doc_rows(self) -> list[EvidenceChunkAudit]:
        return [row for row in self.rows if row.chunks_on_page == 0]

    def token_percentiles(self) -> dict[str, float]:
        if len(self.token_counts) < 2:
            return {}
        p = quantiles(self.token_counts, n=100, method="inclusive")
        return {"p5": p[4], "p50": p[49], "p95": p[94], "max": float(max(self.token_counts))}


class ChunkAuditor:
    def __init__(self, chunks_by_doc: Mapping[str, Sequence[Chunk]], min_coverage: float) -> None:
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be within [0, 1]")
        self._chunks_by_doc = chunks_by_doc
        self._min_coverage = min_coverage

    def audit(self, items: Iterable[GoldenItem]) -> ChunkAuditReport:
        rows: list[EvidenceChunkAudit] = []
        for item in items:
            for span in item.evidence:
                on_page = [c for c in self._chunks_by_doc.get(span.doc_id, ()) if covers(c, span)]
                row = EvidenceChunkAudit(
                    item_id=item.item_id,
                    doc_id=span.doc_id,
                    page_index=span.page_index,
                    chunks_on_page=len(on_page),
                )
                if on_page and audit_tokens(span.text):
                    scored = [(token_coverage(span.text, c.text), c.chunk_id) for c in on_page]
                    coverage, chunk_id = max(scored)
                    row = row.model_copy(
                        update={"best_chunk_id": chunk_id, "best_coverage": coverage}
                    )
                rows.append(row)
        all_chunks = [c for chunks in self._chunks_by_doc.values() for c in chunks]
        return ChunkAuditReport(
            min_coverage=self._min_coverage,
            rows=rows,
            chunk_count=len(all_chunks),
            token_counts=[int(c.metadata.get("token_count", 0)) for c in all_chunks],
        )
