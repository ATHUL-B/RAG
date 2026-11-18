"""Local sentence-transformers embedder (ADR 0003, HLD 004)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Final

import numpy as np

from rag.errors import IndexingError
from rag.indexing.base import Embedder, Matrix, Vector

if TYPE_CHECKING:  # heavy import (PyTorch): only load when the model is actually used
    from sentence_transformers import SentenceTransformer

AUTO_DEVICE: Final = "auto"


def resolve_device(requested: str) -> str:
    """``auto`` picks Apple MPS, then CUDA, then CPU."""
    if requested != AUTO_DEVICE:
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers bi-encoder with L2-normalized float32 output.

    The model is loaded lazily on first use, so building the object (e.g. to compute
    an index path) is cheap and offline.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        query_prefix: str = "",
        batch_size: int = 64,
        device: str = AUTO_DEVICE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._model_name = model_name
        self._revision = revision
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._device = device

    @property
    def model_id(self) -> str:
        return f"{self._model_name}@{self._revision}" if self._revision else self._model_name

    @cached_property
    def _model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(
                self._model_name, revision=self._revision, device=resolve_device(self._device)
            )
        except OSError as exc:
            raise IndexingError(f"cannot load embedding model {self.model_id}: {exc}") from exc

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise IndexingError(f"{self.model_id} does not report an embedding dimension")
        return int(dimension)

    @property
    def max_tokens(self) -> int:
        limit = self._model.max_seq_length
        if limit is None:
            raise IndexingError(f"{self.model_id} does not report a max sequence length")
        return int(limit)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._encode([self._query_prefix + text])[0]
        return vector

    def _encode(self, texts: list[str]) -> Matrix:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
