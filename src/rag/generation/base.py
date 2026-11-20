"""Context assembly and generation contracts (study guide, Part 11; HLD 010)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from rag.domain import Answer, Query, ScoredChunk


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    user: str
    chunk_ids: tuple[str, ...]
    """Which chunks made it into the prompt: ladder rung 4 in the eval."""
    source_labels: dict[str, str] = Field(default_factory=dict)
    """Citation label shown to the model (e.g. "S1") → chunk_id."""
    response_schema: dict[str, Any] | None = None
    """JSON schema the answer must follow (structured output), if any."""


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model_id: str
    """The model that actually produced the text (may differ after a fallback)."""
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    """Includes thinking tokens where the provider bills them as output."""
    stop_reason: str = "end_turn"
    refused: bool = False
    """The provider declined to answer (safety refusal), as opposed to "not answerable"."""


class GenerationResult(BaseModel):
    """An answer plus everything the evaluation needs to judge how it was produced."""

    model_config = ConfigDict(frozen=True)

    answer: Answer
    completion: Completion
    prompt_chunk_ids: tuple[str, ...]
    invalid_citations: tuple[str, ...] = ()
    """Citation labels in the answer that match no source shown (e.g. "S9")."""
    working: str = ""
    """The model's worked steps (line items, values, formula), shown before the answer."""


class LLMClient(ABC):
    """Provider-agnostic LLM call; providers are chosen in ADR 0005."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def complete(self, prompt: Prompt) -> Completion: ...


class ContextAssembler(ABC):
    """Orders, labels and budgets the evidence that goes into the prompt."""

    @abstractmethod
    def assemble(self, query: Query, chunks: Sequence[ScoredChunk]) -> Prompt: ...


class AnswerGenerator(ABC):
    """Produces a cited answer, or refuses when the evidence is insufficient."""

    @abstractmethod
    def generate(self, query: Query, chunks: Sequence[ScoredChunk]) -> GenerationResult: ...
