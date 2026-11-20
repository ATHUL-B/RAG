import pytest

from rag.domain import Chunk
from rag.evaluation.generation_runner import AnswerOutcome, EvidenceReach, GenerationTrace
from rag.evaluation.judge_runner import JudgeRunner, rebuild_sources
from rag.evaluation.judges import Judge, JudgeCase, JudgeKind, Verdict


def _chunk(chunk_id: str, page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="3M_2018_10K",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=page,
        page_end=page,
    )


CHUNKS = {
    "a": _chunk("a", 59, "Purchases of PP&E\n \n(1,577)"),
    "b": _chunk("b", 40, "Revenue 32,765"),
}


def test_sources_are_rebuilt_with_the_generators_labels_and_order() -> None:
    assert rebuild_sources(["b", "a"], CHUNKS) == (
        "[S1] 3M_2018_10K, page 40\nRevenue 32,765\n\n"
        "[S2] 3M_2018_10K, page 59\nPurchases of PP&E (1,577)"
    )
    assert rebuild_sources(["missing"], CHUNKS) == ""


class Fixed(Judge):
    """Passes answers containing ``pass``; fails on ``boom``."""

    def __init__(self, kind: JudgeKind) -> None:
        self.kind = kind
        self.cases: list[JudgeCase] = []

    def judge(self, case: JudgeCase) -> Verdict | None:
        self.cases.append(case)
        if "boom" in case.answer:
            return None
        label = (
            ("correct" if self.kind is JudgeKind.CORRECTNESS else "faithful")
            if "pass" in case.answer
            else ("incorrect" if self.kind is JudgeKind.CORRECTNESS else "unfaithful")
        )
        return Verdict(kind=self.kind, label=label, reason="r", judge_model="m")


def _trace(item: str, answer: str, **kw: object) -> GenerationTrace:
    fields: dict[str, object] = {
        "item_id": item,
        "question": "q",
        "reference_answer": "r",
        "tags": [],
        "evidence_units": [],
        "prompt_chunk_ids": ["a"],
        "answer": answer,
        "evidence_reach": EvidenceReach.IN_PROMPT,
        "outcome": AnswerOutcome.UNSCORED,
    }
    return GenerationTrace.model_validate(fields | kw)


def test_judges_answers_skips_errors_and_refusals_and_summarises() -> None:
    correctness, faithfulness = Fixed(JudgeKind.CORRECTNESS), Fixed(JudgeKind.FAITHFULNESS)
    traces = [
        _trace("t1", "pass [S1]", numeric_match=True, outcome=AnswerOutcome.CORRECT),
        _trace("t2", "nope", numeric_match=True, outcome=AnswerOutcome.CORRECT),
        _trace("t3", "pass", evidence_reach=EvidenceReach.NOT_RETRIEVED),
        _trace("t4", "boom"),
        _trace("err", "", outcome=AnswerOutcome.ERROR, error="bad json"),
        _trace("ref", "Not in the sources.", refused=True, outcome=AnswerOutcome.REFUSED),
    ]
    judged, summary = JudgeRunner([correctness, faithfulness], CHUNKS).run(traces)

    assert [a.item_id for a in judged] == ["t1", "t2", "t3", "t4"]
    assert judged[3].verdicts == {}  # both judges failed on t4
    assert summary.skipped == {"error": 1, "refused": 1}
    assert summary.failures == {"correctness": 1, "faithfulness": 1}
    assert summary.rates == {
        "correctness": pytest.approx(2 / 3),
        "faithfulness": pytest.approx(2 / 3),
    }
    assert summary.by_reach["not_retrieved"] == {"n": 1.0, "correctness": 1.0, "faithfulness": 1.0}
    assert summary.by_reach["in_prompt"]["n"] == 3.0
    # t1: judge passes, numeric passes → agree; t2: judge fails, numeric passes → disagree
    assert summary.numeric_agreement == {"n": 2.0, "agreement": 0.5}
    assert correctness.cases[0].sources.startswith("[S1] 3M_2018_10K, page 59")


def test_requires_a_judge() -> None:
    with pytest.raises(ValueError, match="at least one judge"):
        JudgeRunner([], CHUNKS)
