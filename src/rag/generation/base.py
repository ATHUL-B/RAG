"""Context assembly and generation contracts (study guide, Part 11)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from rag.domain import Answer, Query, ScoredChunk


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    user: str
    chunk_ids: tuple[str, ...]
    """Which chunks made it into the prompt: ladder rung 4 in the eval."""


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model_id: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class LLMClient(ABC):
    """Provider-agnostic LLM call; the provider is chosen in an ADR."""

    @abstractmethod
    def complete(self, prompt: Prompt) -> Completion: ...


class ContextAssembler(ABC):
    """Orders, labels and budgets the evidence that goes into the prompt."""

    @abstractmethod
    def assemble(self, query: Query, chunks: Sequence[ScoredChunk]) -> Prompt: ...


class AnswerGenerator(ABC):
    """Produces a cited answer, or refuses when the evidence is insufficient."""

    @abstractmethod
    def generate(self, query: Query, chunks: Sequence[ScoredChunk]) -> Answer: ...
