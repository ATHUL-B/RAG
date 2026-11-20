"""Grounded answer generation: assemble → complete → parse → validate citations (HLD 010)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from rag.domain import Answer, Query, ScoredChunk
from rag.errors import UnusableAnswerError
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
_MARKUP = re.compile(
    r"<\s*/?\s*(?:parameter|invoke|function_calls|antml:[a-z_]+)\b|<!\[CDATA\[|<br\s*/?>",
    re.IGNORECASE,
)
"""Tool-call or transport markup leaking into the answer text (seen in the first full run)."""

SCHEMA_MISMATCH = "schema_mismatch"
EMPTY_ANSWER = "empty_answer"
BARE_BOOLEAN = "bare_boolean"
MARKUP_LEAK = "markup_leak"


def answer_problem(text: str) -> str | None:
    """Why ``text`` is not a usable answer, or None if it is.

    Passing the JSON schema only proves the *shape*: ``{"answerable": true, "answer": ""}``
    is valid JSON but useless. Both answers and refusals must say something, because a
    refusal should explain what is missing (Part 11).
    """
    stripped = text.strip()
    if not stripped:
        return EMPTY_ANSWER
    if stripped.lower() in {"true", "false"}:
        return BARE_BOOLEAN
    if _MARKUP.search(stripped):
        return MARKUP_LEAK
    return None


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
    """Read ``{"answerable", "answer"}`` and check the answer is usable.

    Plain text is accepted when no schema was set. Every failure raises
    ``UnusableAnswerError`` with the raw output, so it is counted and inspectable, never
    silently scored as an answer or a refusal.
    """
    if prompt.response_schema is None:
        answerable, text = True, completion.text.strip()
    else:
        try:
            data = json.loads(completion.text)
            answerable, text = bool(data["answerable"]), str(data["answer"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise _unusable(completion, prompt, SCHEMA_MISMATCH) from exc
    problem = answer_problem(text)
    if problem is not None:
        raise _unusable(completion, prompt, problem)
    return answerable, text


def _unusable(completion: Completion, prompt: Prompt, kind: str) -> UnusableAnswerError:
    message = (
        f"{completion.model_id} returned an unusable answer ({kind}, "
        f"stop_reason={completion.stop_reason}): {completion.text[:200]!r}"
    )
    return UnusableAnswerError(
        message, kind=kind, raw_output=completion.text, prompt_chunk_ids=prompt.chunk_ids
    )
