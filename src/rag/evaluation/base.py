"""Evaluation contracts (study guide, Part 13).

Retrieval metrics are pure functions (added with the eval-harness milestone).
Generation is judged by a :class:`Judge`, which may be an LLM and must therefore
be calibrated against human labels before its scores are trusted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import Answer, GoldenItem, ScoredChunk


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class Judge(ABC):
    @property
    @abstractmethod
    def metric(self) -> str:
        """Name of what is judged, e.g. 'faithfulness' or 'correctness'."""

    @abstractmethod
    def judge(
        self, item: GoldenItem, answer: Answer, context: Sequence[ScoredChunk]
    ) -> Verdict: ...
