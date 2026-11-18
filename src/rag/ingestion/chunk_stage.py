"""Load parsed documents → chunk → save one ChunkSet per document (HLD 003)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from rag.domain import ChunkSet, Document
from rag.errors import IngestionError, StorageError
from rag.ingestion.base import Chunker
from rag.storage import JsonArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class ChunkReport:
    chunked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0


class ChunkStage:
    def __init__(
        self,
        chunker: Chunker,
        documents: JsonArtifactStore[Document],
        chunk_sets: JsonArtifactStore[ChunkSet],
    ) -> None:
        self._chunker = chunker
        self._documents = documents
        self._chunk_sets = chunk_sets

    def run(self, doc_ids: Iterable[str], force: bool = False) -> ChunkReport:
        report = ChunkReport()
        for doc_id in doc_ids:
            if not force and self._chunk_sets.exists(doc_id):
                report.skipped.append(doc_id)
                continue
            try:
                chunks = self._chunker.chunk(self._documents.load(doc_id))
                chunk_set = ChunkSet(
                    doc_id=doc_id,
                    chunker_fingerprint=self._chunker.fingerprint,
                    chunks=tuple(chunks),
                )
                self._chunk_sets.save(doc_id, chunk_set)
            except (IngestionError, StorageError) as exc:
                logger.error("chunking failed", extra={"doc_id": doc_id, "error": str(exc)})
                report.failed[doc_id] = str(exc)
                continue
            report.chunked.append(doc_id)
            report.chunks += len(chunks)
        return report
