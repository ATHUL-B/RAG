"""Parse → enrich → save, for many documents (HLD 002).

Idempotent and failure-tolerant: re-running skips documents that already have an
artifact, and one bad PDF is recorded without stopping the batch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rag.domain import Document
from rag.errors import IngestionError, StorageError
from rag.ingestion.base import MetadataEnricher, Parser
from rag.observability import StageTimer
from rag.observability.timing import Clock
from rag.storage import JsonArtifactStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    doc_id: str
    path: Path


@dataclass
class ParseReport:
    parsed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    pages: int = 0
    failed_pages: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    """Parse + enrich + save time per parsed document."""


class ParseStage:
    def __init__(
        self,
        parser: Parser,
        enricher: MetadataEnricher,
        store: JsonArtifactStore[Document],
        clock: Clock = time.perf_counter,
    ) -> None:
        self._parser = parser
        self._enricher = enricher
        self._store = store
        self._clock = clock

    def run(self, sources: Iterable[SourceFile], force: bool = False) -> ParseReport:
        report = ParseReport()
        for source in sources:
            if not force and self._store.exists(source.doc_id):
                report.skipped.append(source.doc_id)
                continue
            timer = StageTimer(self._clock)
            try:
                with timer.stage(source.doc_id):
                    document = self._enricher.enrich(self._parser.parse(source.path, source.doc_id))
                    self._store.save(source.doc_id, document)
            except (IngestionError, StorageError) as exc:
                logger.error("parse failed", extra={"doc_id": source.doc_id, "error": str(exc)})
                report.failed[source.doc_id] = str(exc)
                continue
            report.parsed.append(source.doc_id)
            report.pages += len(document.pages)
            report.failed_pages += int(document.metadata.get("failed_pages", 0))
            report.timings_ms.update(timer.timings_ms)
            logger.info(
                "parsed",
                extra={
                    "doc_id": source.doc_id,
                    "pages": len(document.pages),
                    "ms": round(timer.total_ms, 1),
                },
            )
        return report
