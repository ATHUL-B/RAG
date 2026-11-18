"""Cross-encoder reranking (study guide, Part 10; ADR 0004; HLD 008).

A bi-encoder (our embedder) encodes question and chunk *separately*, so chunk vectors can
be precomputed but the two texts never "see" each other. A cross-encoder reads them
*together* in one forward pass: much better at judging "does this chunk answer this
question?", but too slow to run over the whole corpus, so it only rescores a shortlist.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Final, Protocol

from rag.domain import Query, ScoredChunk
from rag.errors import RetrievalError
from rag.indexing.embedders import AUTO_DEVICE, resolve_device
from rag.retrieval.base import Reranker

if TYPE_CHECKING:  # heavy import (PyTorch): only when the model is actually used
    from sentence_transformers import CrossEncoder

_STAGE: Final = "rerank"


class PairScorer(Protocol):
    """Scores (question, passage) pairs; higher = more relevant."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class SentenceTransformersPairScorer:
    """sentence-transformers ``CrossEncoder``, lazily loaded, revision pinned."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = AUTO_DEVICE,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be >= 1")
        self._model_name = model_name
        self._revision = revision
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

    @property
    def model_id(self) -> str:
        return f"{self._model_name}@{self._revision}" if self._revision else self._model_name

    @cached_property
    def _model(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        try:
            model: CrossEncoder = CrossEncoder(
                self._model_name,
                revision=self._revision,
                max_length=self._max_length,
                device=resolve_device(self._device),
            )
        except OSError as exc:
            raise RetrievalError(f"cannot load reranker {self.model_id}: {exc}") from exc
        return model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            list(pairs), batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


class CrossEncoderReranker(Reranker):
    def __init__(self, scorer: PairScorer) -> None:
        self._scorer = scorer

    def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Rescore ``candidates`` against the question; ties keep first-stage order."""
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not candidates:
            return []
        scores = self._scorer.score([(query.text, c.chunk.text) for c in candidates])
        if len(scores) != len(candidates):
            raise RetrievalError(f"scorer returned {len(scores)} scores for {len(candidates)}")
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:top_n]
        return [
            ScoredChunk(chunk=candidates[i].chunk, score=scores[i], rank=rank, stage=_STAGE)
            for rank, i in enumerate(order, start=1)
        ]
