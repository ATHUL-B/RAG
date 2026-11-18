"""Online retrieval contracts (study guide, Parts 6-10)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.domain import Query, ScoredChunk


class QueryRewriter(ABC):
    """Turns a user question into one or more search queries (Part 7)."""

    @abstractmethod
    def rewrite(self, query: Query) -> list[Query]: ...


class Retriever(ABC):
    """First-stage candidate generation: cheap, wide, recall-oriented."""

    @abstractmethod
    def retrieve(self, query: Query, k: int) -> list[ScoredChunk]: ...


class Fuser(ABC):
    """Merges several ranked lists into one, e.g. Reciprocal Rank Fusion (Part 9)."""

    @abstractmethod
    def fuse(self, rankings: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]: ...


class Reranker(ABC):
    """Second stage: reads query and chunk together, precision-oriented (Part 10)."""

    @abstractmethod
    def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...
