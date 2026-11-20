"""Online retrieval contracts (study guide, Parts 6-10)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import Filters, Query, ScoredChunk


class RetrievalResult(BaseModel):
    """Candidates plus what the retriever actually did, so traces can explain them."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[ScoredChunk, ...]
    applied_filters: Filters = Field(default_factory=dict)
    """Filters in force for the final search (after any relaxation)."""
    relaxed: tuple[str, ...] = ()
    """Filter keys dropped, in order, because the stricter search found too little."""
    stage_candidates: dict[str, tuple[ScoredChunk, ...]] = Field(default_factory=dict)
    """Intermediate ranked lists by stage name (e.g. "dense", "bm25" before fusion)."""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    """Wall-clock milliseconds of internal stages (e.g. "first_stage", "rerank")."""
    queries: tuple[str, ...] = ()
    """The search texts actually used, when a rewriter changed them (Part 7)."""


class QueryRewriter(ABC):
    """Turns a user question into one or more search queries (Part 7)."""

    @abstractmethod
    def rewrite(self, query: Query) -> list[Query]: ...


class FilterExtractor(ABC):
    """Reads metadata constraints (company, year, ...) out of a question (Parts 7-8)."""

    @abstractmethod
    def extract(self, text: str) -> Filters: ...


class Retriever(ABC):
    """First-stage candidate generation: cheap, wide, recall-oriented."""

    @abstractmethod
    def retrieve(self, query: Query, k: int) -> RetrievalResult: ...


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
