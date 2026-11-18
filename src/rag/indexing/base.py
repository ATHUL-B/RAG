"""Embedding and index contracts (study guide, Part 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.domain import Chunk, Metadata, ScoredChunk

Vector = Sequence[float]


class Embedder(ABC):
    """Bi-encoder: embeds queries and chunks separately so chunks can be precomputed."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Exact model name + version; stored on every vector (CLAUDE.md §2)."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Separate from documents: many models use a different prefix for queries."""


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    @abstractmethod
    def search(self, vector: Vector, k: int, filters: Metadata | None = None) -> list[ScoredChunk]:
        """Top-k by similarity, applying ``filters`` *before* ranking (Part 8)."""


class KeywordIndex(ABC):
    """Lexical (e.g. BM25) index for exact terms the embedder blurs (Part 9)."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def search(self, text: str, k: int, filters: Metadata | None = None) -> list[ScoredChunk]: ...
