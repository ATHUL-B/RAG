"""Deterministic, offline stand-in for a real embedding model."""

import zlib
from collections.abc import Sequence

import numpy as np

from rag.indexing.base import Embedder, Matrix, Vector


class HashingEmbedder(Embedder):
    """Hashed bag-of-words, L2-normalized: texts sharing words get similar vectors."""

    def __init__(self, dimension: int = 64, max_tokens: int = 512) -> None:
        self._dimension = dimension
        self._max_tokens = max_tokens
        self.document_calls = 0

    @property
    def model_id(self) -> str:
        return f"hashing-bow-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.document_calls += 1
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % self._dimension] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0] = 1.0  # empty text: arbitrary but valid unit vector
            return vector
        return vector / norm
