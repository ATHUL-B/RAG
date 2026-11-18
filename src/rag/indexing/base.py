"""Embedding and index contracts (study guide, Part 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from rag.domain import Chunk, Filters, ScoredChunk

Vector = npt.NDArray[np.float32]
"""One embedding, shape ``(dimension,)``."""

Matrix = npt.NDArray[np.float32]
"""Many embeddings, shape ``(n, dimension)``; row i belongs to input i."""


class Embedder(ABC):
    """Bi-encoder: embeds queries and chunks separately so chunks can be precomputed."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Exact model name + revision; stored with every index (CLAUDE.md §2)."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Longer inputs are truncated by the model: their tail is never embedded."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Length in the *model's own* tokenizer (may differ from the chunker's)."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Separate from documents: many models use a different prefix for queries."""


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None: ...

    @abstractmethod
    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        """Top-k by similarity, applying ``filters`` *before* ranking (Part 8)."""


class KeywordIndex(ABC):
    """Lexical (e.g. BM25) index for exact terms the embedder blurs (Part 9)."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def search(self, text: str, k: int, filters: Filters | None = None) -> list[ScoredChunk]: ...
