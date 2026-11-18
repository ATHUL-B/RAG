"""Embed every chunk set and build the exact vector index (HLD 004)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from rag.domain import Chunk, ChunkSet
from rag.indexing.base import Embedder
from rag.indexing.vector_store import ExactVectorStore
from rag.observability.timing import Clock
from rag.storage import JsonArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    documents: int = 0
    chunks: int = 0
    truncated_chunk_ids: list[str] = field(default_factory=list)
    """Chunks longer than the model's limit: their tail is not embedded."""
    seconds: float = 0.0

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.seconds if self.seconds else 0.0


class IndexStage:
    def __init__(
        self,
        embedder: Embedder,
        chunk_sets: JsonArtifactStore[ChunkSet],
        batch_size: int = 512,
        clock: Clock = time.perf_counter,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._embedder = embedder
        self._chunk_sets = chunk_sets
        self._batch_size = batch_size
        self._clock = clock

    def build(self, doc_ids: Iterable[str]) -> tuple[ExactVectorStore, IndexReport]:
        store = ExactVectorStore(self._embedder.model_id, self._embedder.dimension)
        report = IndexReport()
        started = self._clock()
        batch: list[Chunk] = []
        for doc_id in doc_ids:
            report.documents += 1
            for chunk in self._chunk_sets.load(doc_id).chunks:
                batch.append(chunk)
                if len(batch) >= self._batch_size:
                    self._embed_batch(batch, store, report)
                    batch = []
        if batch:
            self._embed_batch(batch, store, report)
        report.seconds = self._clock() - started
        return store, report

    def _embed_batch(
        self, batch: list[Chunk], store: ExactVectorStore, report: IndexReport
    ) -> None:
        limit = self._embedder.max_tokens
        report.truncated_chunk_ids.extend(
            c.chunk_id for c in batch if self._embedder.count_tokens(c.text) > limit
        )
        store.upsert(batch, self._embedder.embed_documents([c.text for c in batch]))
        report.chunks += len(batch)
        logger.info("embedded batch", extra={"chunks_total": report.chunks})
