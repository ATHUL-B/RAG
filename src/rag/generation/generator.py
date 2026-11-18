"""Grounded answer generation: assemble → complete → parse → validate citations (HLD 010)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from rag.domain import Answer, Query, ScoredChunk
from rag.errors import GenerationError
from rag.generation.base import (
    AnswerGenerator,
    Completion,
    ContextAssembler,
    GenerationResult,
    LLMClient,
    Prompt,
)

_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
_LABEL = re.compile(r"S\d+")


def extract_citations(text: str) -> list[str]:
    """Citation labels in order of first appearance: "[S2] … [S1, S3]" → S2, S1, S3.

    Scans bracket groups left to right; a group counts only if every comma-separated part is
    a label, so "[in millions]" or "[1]" are ignored.
    """
    labels: list[str] = []
    for group in _BRACKETS.findall(text):
        parts = [part.strip() for part in group.split(",")]
        if parts and all(_LABEL.fullmatch(part) for part in parts):
            labels.extend(parts)
    return list(dict.fromkeys(labels))


class GroundedAnswerGenerator(AnswerGenerator):
    def __init__(self, assembler: ContextAssembler, llm: LLMClient) -> None:
        self._assembler = assembler
        self._llm = llm

    def generate(self, query: Query, chunks: Sequence[ScoredChunk]) -> GenerationResult:
        prompt = self._assembler.assemble(query, chunks)
        completion = self._llm.complete(prompt)
        if completion.refused:
            answer = Answer(text="", refused=True)
            return GenerationResult(
                answer=answer, completion=completion, prompt_chunk_ids=prompt.chunk_ids
            )

        answerable, text = _parse(completion, prompt)
        labels = extract_citations(text)
        cited = tuple(
            prompt.source_labels[label] for label in labels if label in prompt.source_labels
        )
        invalid = tuple(label for label in labels if label not in prompt.source_labels)
        return GenerationResult(
            answer=Answer(text=text, cited_chunk_ids=cited, refused=not answerable),
            completion=completion,
            prompt_chunk_ids=prompt.chunk_ids,
            invalid_citations=invalid,
        )


def _parse(completion: Completion, prompt: Prompt) -> tuple[bool, str]:
    """Read ``{"answerable", "answer"}``; plain text is accepted when no schema was set."""
    if prompt.response_schema is None:
        return True, completion.text.strip()
    try:
        data = json.loads(completion.text)
        return bool(data["answerable"]), str(data["answer"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GenerationError(
            f"{completion.model_id} returned output that does not match the answer schema "
            f"(stop_reason={completion.stop_reason}): {completion.text[:200]!r}"
        ) from exc
