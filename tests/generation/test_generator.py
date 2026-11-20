import json
from collections.abc import Sequence

import pytest

from rag.domain import Chunk, Query, ScoredChunk
from rag.errors import UnusableAnswerError
from rag.generation.base import Completion, GenerationResult, LLMClient, Prompt
from rag.generation.context import SourcesContextAssembler
from rag.generation.generator import (
    BARE_BOOLEAN,
    EMPTY_ANSWER,
    MARKUP_LEAK,
    SCHEMA_MISMATCH,
    GroundedAnswerGenerator,
    answer_problem,
    extract_citations,
)


class ScriptedLLM(LLMClient):
    def __init__(self, text: str = "", refused: bool = False) -> None:
        self._text, self._refused = text, refused
        self.prompts: list[Prompt] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        return Completion(
            text=self._text,
            model_id="fake",
            input_tokens=10,
            output_tokens=5,
            refused=self._refused,
        )


def _candidates(n: int = 3) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                doc_id="D",
                text=f"text {i}",
                char_start=0,
                char_end=6,
                page_start=i,
                page_end=i,
            ),
            score=1.0,
            rank=i,
            stage="dense",
        )
        for i in range(1, n + 1)
    ]


def _generate(llm: LLMClient) -> GenerationResult:
    generator = GroundedAnswerGenerator(SourcesContextAssembler(max_sources=3), llm)
    return generator.generate(Query(text="q"), _candidates())


def _json(answerable: bool, answer: str) -> str:
    return json.dumps({"answerable": answerable, "answer": answer})


@pytest.mark.parametrize(
    ("text", "labels"),
    [
        ("Capex was $1,577M [S2]. It fell [S1][S3].", ["S2", "S1", "S3"]),
        ("Both [S1, S3] agree [S1].", ["S1", "S3"]),
        ("Units [in millions] and footnote [1] are not citations.", []),
        ("S10 then S1: [S10] [S1]", ["S10", "S1"]),
    ],
)
def test_extract_citations(text: str, labels: list[str]) -> None:
    assert extract_citations(text) == labels


def test_answer_citations_map_to_chunks_and_invalid_labels_are_kept() -> None:
    result = _generate(ScriptedLLM(_json(True, "Capex $1,577M [S2]; see also [S9].")))
    assert result.answer.text == "Capex $1,577M [S2]; see also [S9]."
    assert result.answer.cited_chunk_ids == ("c2",)
    assert result.invalid_citations == ("S9",)
    assert result.prompt_chunk_ids == ("c1", "c2", "c3")
    assert not result.answer.refused


def test_not_answerable_is_a_refusal() -> None:
    result = _generate(ScriptedLLM(_json(False, "The sources lack FY2018 capex.")))
    assert result.answer.refused
    assert result.answer.text == "The sources lack FY2018 capex."


def test_provider_refusal_is_a_refusal_without_parsing() -> None:
    result = _generate(ScriptedLLM("", refused=True))
    assert result.answer.refused
    assert result.completion.refused


def test_output_that_breaks_the_schema_is_an_error() -> None:
    with pytest.raises(UnusableAnswerError, match="schema_mismatch") as info:
        _generate(ScriptedLLM("Capex was 1577 [S1]"))
    assert info.value.kind == SCHEMA_MISMATCH
    assert info.value.raw_output == "Capex was 1577 [S1]"
    assert info.value.prompt_chunk_ids == ("c1", "c2", "c3")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", EMPTY_ANSWER),
        ("  \n ", EMPTY_ANSWER),
        ("true", BARE_BOOLEAN),
        (" False ", BARE_BOOLEAN),
        ('<parameter name="answerable">false', MARKUP_LEAK),
        ("<br>", MARKUP_LEAK),
        ("<![CDATA[The margin is 2.8% [S1].", MARKUP_LEAK),
        ("</invoke>", MARKUP_LEAK),
        ("Capex was $1,577 million [S2].", None),
        ("Yes. Boeing reports two product categories [S1].", None),
        ("The ratio fell (x < y) [S1].", None),  # a plain "<" is not markup
    ],
)
def test_answer_problem(text: str, problem: str | None) -> None:
    assert answer_problem(text) == problem


@pytest.mark.parametrize(
    ("answerable", "answer", "kind"),
    [
        (True, "", EMPTY_ANSWER),  # first full run: 23 "answers" were empty
        (False, "", EMPTY_ANSWER),  # a refusal must say what is missing
        (True, "true", BARE_BOOLEAN),
        (False, '<parameter name="answerable">false', MARKUP_LEAK),
    ],
)
def test_schema_valid_but_unusable_answers_are_errors(
    answerable: bool, answer: str, kind: str
) -> None:
    raw = _json(answerable, answer)
    with pytest.raises(UnusableAnswerError) as info:
        _generate(ScriptedLLM(raw))
    assert info.value.kind == kind
    assert info.value.raw_output == raw


def test_unusable_plain_text_without_a_schema_is_an_error() -> None:
    class NoSchema(SourcesContextAssembler):
        def assemble(self, query: Query, chunks: Sequence[ScoredChunk]) -> Prompt:
            return super().assemble(query, chunks).model_copy(update={"response_schema": None})

    generator = GroundedAnswerGenerator(NoSchema(), ScriptedLLM("   "))
    with pytest.raises(UnusableAnswerError, match=EMPTY_ANSWER):
        generator.generate(Query(text="q"), _candidates())


def test_plain_text_is_accepted_without_a_schema() -> None:
    class NoSchema(SourcesContextAssembler):
        def assemble(self, query: Query, chunks: Sequence[ScoredChunk]) -> Prompt:
            return super().assemble(query, chunks).model_copy(update={"response_schema": None})

    generator = GroundedAnswerGenerator(NoSchema(), ScriptedLLM("Capex was 1577 [S1]"))
    result = generator.generate(Query(text="q"), _candidates())
    assert result.answer.cited_chunk_ids == ("c1",)
