from collections.abc import Sequence

import pytest

from rag.domain import Answer, Chunk, GoldenItem, Query, ScoredChunk, SourceSpan
from rag.errors import GenerationError, UnusableAnswerError
from rag.evaluation.generation_runner import (
    AnswerOutcome,
    EvidenceReach,
    GenerationEvalRunner,
    GenerationRunResult,
    Pricing,
    estimate_cost,
)
from rag.generation.base import AnswerGenerator, Completion, GenerationResult, Prompt
from rag.retrieval.base import RetrievalResult, Retriever
from support.tokenizers import WordCounter

PRICING = Pricing(input_per_mtok=4.0, output_per_mtok=20.0)


def _chunk(chunk_id: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text="t",
        char_start=0,
        char_end=1,
        page_start=page,
        page_end=page,
    )


class FixedRetriever(Retriever):
    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        hits = (_chunk("evidence", 5), _chunk("other", 9))
        return RetrievalResult(
            candidates=tuple(
                ScoredChunk(chunk=c, score=1.0, rank=i, stage="dense")
                for i, c in enumerate(hits, 1)
            )
        )


class ScriptedGenerator(AnswerGenerator):
    """Answers keyed by question text."""

    def __init__(self, answers: dict[str, Answer | Exception]) -> None:
        self._answers = answers

    def generate(self, query: Query, chunks: Sequence[ScoredChunk]) -> GenerationResult:
        answer = self._answers[query.text]
        if isinstance(answer, Exception):
            raise answer
        return GenerationResult(
            answer=answer,
            completion=Completion(text="{}", model_id="m", input_tokens=1000, output_tokens=500),
            prompt_chunk_ids=tuple(c.chunk.chunk_id for c in chunks),
            invalid_citations=("S9",) if "S9" in answer.text else (),
        )


def _item(item_id: str, reference: str, page: int = 5) -> GoldenItem:
    return GoldenItem(
        item_id=item_id,
        question=item_id,
        reference_answer=reference,
        evidence=(SourceSpan(doc_id="D", page_index=page),),
        tags=frozenset({"type:x"}),
    )


ANSWERS: dict[str, Answer | Exception] = {
    "good": Answer(text="Capex was $1,577 million [S1].", cited_chunk_ids=("evidence",)),
    "wrong-cite": Answer(text="Capex was 1,373 [S2] [S9].", cited_chunk_ids=("other",)),
    "refused": Answer(text="Not in the sources.", refused=True),
    "text": Answer(text="Margins improved [S1].", cited_chunk_ids=("evidence",)),
    "broken": GenerationError("bad json"),
}


def _run() -> GenerationRunResult:
    runner = GenerationEvalRunner(FixedRetriever(), ScriptedGenerator(ANSWERS), PRICING, depth=2)
    return runner.run(
        [
            _item("good", "$1577.00"),
            _item("wrong-cite", "$1577.00"),
            _item("refused", "$1577.00"),
            _item("text", "Margins improved."),
            _item("broken", "x"),
        ]
    )


def test_per_question_checks() -> None:
    traces = {t.item_id: t for t in _run().traces}
    good, wrong, refused, text = (traces[k] for k in ("good", "wrong-cite", "refused", "text"))

    assert good.metrics == {
        "answered": 1.0,
        "evidence_in_prompt": 1.0,
        "citation_valid": 1.0,
        "has_citation": 1.0,
        "cites_evidence": 1.0,
    }
    assert good.numeric_match is True
    assert wrong.metrics["citation_valid"] == 0.0 and wrong.invalid_citations == ["S9"]
    assert wrong.metrics["cites_evidence"] == 0.0 and wrong.numeric_match is False
    assert refused.metrics["answered"] == 0.0 and refused.numeric_match is False
    assert text.numeric_match is None  # reference is not a plain number
    assert good.cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)


def test_errors_are_recorded_not_raised_and_totals_add_up() -> None:
    result = _run()
    broken = next(t for t in result.traces if t.item_id == "broken")
    assert broken.error == "bad json"
    assert result.errors == 1
    assert result.report is not None
    assert result.report.overall.count == 4
    assert result.numeric == {"n": 3, "matched": 1, "rate": pytest.approx(1 / 3)}
    assert result.totals["input_tokens"] == 4000
    assert result.totals["cost_usd"] == pytest.approx(4 * 0.014)
    assert set(result.latency_ms) == {"retrieve", "generate"}


def test_estimate_cost_uses_prompt_tokens_margin_and_expected_output() -> None:
    prompts = [Prompt(system="a b", user="c d e f", chunk_ids=())] * 2  # 6 words each
    cost = estimate_cost(
        prompts, WordCounter(), PRICING, expected_output_tokens=100, safety_margin=1.5
    )
    assert cost == pytest.approx((18 * 4 + 200 * 20) / 1e6)


def test_rejects_empty_runs_and_bad_depth() -> None:
    runner = GenerationEvalRunner(FixedRetriever(), ScriptedGenerator({}), PRICING)
    with pytest.raises(ValueError, match="no items"):
        runner.run([])
    with pytest.raises(ValueError, match="depth"):
        GenerationEvalRunner(FixedRetriever(), ScriptedGenerator({}), PRICING, depth=0)


class ShowsOnlyOther(ScriptedGenerator):
    """Context assembly that drops the evidence chunk: only "other" reaches the prompt."""

    def generate(self, query: Query, chunks: Sequence[ScoredChunk]) -> GenerationResult:
        return super().generate(query, [c for c in chunks if c.chunk.chunk_id == "other"])


def test_ladder_places_each_question_by_evidence_reach_and_outcome() -> None:
    traces = {t.item_id: t for t in _run().traces}
    assert {k: (t.evidence_reach, t.outcome) for k, t in traces.items()} == {
        "good": (EvidenceReach.IN_PROMPT, AnswerOutcome.CORRECT),
        "wrong-cite": (EvidenceReach.IN_PROMPT, AnswerOutcome.WRONG),
        "refused": (EvidenceReach.IN_PROMPT, AnswerOutcome.REFUSED),
        "text": (EvidenceReach.IN_PROMPT, AnswerOutcome.UNSCORED),
        "broken": (None, AnswerOutcome.ERROR),  # API error: no prompt, reach unknown
    }
    assert traces["good"].raw_output == "{}"
    assert traces["broken"].error_kind == "api_error"
    assert _run().ladder == {
        "in_prompt": {"correct": 1, "wrong": 1, "refused": 1, "unscored": 1},
        "unknown": {"error": 1},
    }


def test_evidence_lost_before_the_prompt_is_attributed_to_the_right_rung() -> None:
    answers: dict[str, Answer | Exception] = {
        q: Answer(text="Not in the sources.", refused=True)
        for q in ("unindexed", "unretrieved", "dropped")
    }
    items = [
        _item("unindexed", "$1.00", page=99),
        _item("unretrieved", "$1.00", page=7),
        _item("dropped", "$1.00", page=5),
    ]
    runner = GenerationEvalRunner(
        FixedRetriever(),
        ShowsOnlyOther(answers),
        PRICING,
        depth=2,
        indexed={"D#p5", "D#p7", "D#p9"},
    )
    reach = {t.item_id: t.evidence_reach for t in runner.run(items).traces}
    assert reach == {
        "unindexed": EvidenceReach.NOT_IN_INDEX,
        "unretrieved": EvidenceReach.NOT_RETRIEVED,
        "dropped": EvidenceReach.NOT_IN_PROMPT,
    }


def test_without_index_info_missing_evidence_counts_as_not_retrieved() -> None:
    answers: dict[str, Answer | Exception] = {"q": Answer(text="No.", refused=True)}
    runner = GenerationEvalRunner(FixedRetriever(), ScriptedGenerator(answers), PRICING)
    (trace,) = runner.run([_item("q", "x", page=99)]).traces
    assert trace.evidence_reach is EvidenceReach.NOT_RETRIEVED


def test_unusable_answers_keep_kind_raw_output_and_prompt_reach() -> None:
    raw = '{"answerable": true, "answer": ""}'
    answers: dict[str, Answer | Exception] = {
        "shown": UnusableAnswerError("empty", "empty_answer", raw, ("evidence", "other")),
        "hidden": UnusableAnswerError("leak", "markup_leak", "<br>", ("other",)),
    }
    runner = GenerationEvalRunner(FixedRetriever(), ScriptedGenerator(answers), PRICING, depth=2)
    result = runner.run([_item("shown", "$1.00"), _item("hidden", "$1.00")])
    traces = {t.item_id: t for t in result.traces}

    shown = traces["shown"]
    assert (shown.error_kind, shown.raw_output) == ("empty_answer", raw)
    assert shown.prompt_chunk_ids == ["evidence", "other"]
    assert (shown.evidence_reach, shown.outcome) == (EvidenceReach.IN_PROMPT, AnswerOutcome.ERROR)
    assert traces["hidden"].evidence_reach is EvidenceReach.NOT_IN_PROMPT
    assert result.error_kinds == {"empty_answer": 1, "markup_leak": 1}
    assert result.errors == 2 and result.report is None
