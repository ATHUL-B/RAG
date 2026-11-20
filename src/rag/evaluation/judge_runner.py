"""Judge a finished generation run, offline (HLD 016).

Judging is separate from generating: a run's ``generations.jsonl`` already holds the
question, reference, answer, working and the chunk IDs that were in the prompt. The
sources are rebuilt from the index with the same assembler, so the faithfulness judge sees
exactly the ``[S#]`` labels the generator saw. Re-judging a run (or calibrating a new
judge) never re-runs generation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import Chunk, Query, ScoredChunk
from rag.evaluation.generation_runner import AnswerOutcome, GenerationTrace
from rag.evaluation.judges import Judge, JudgeCase, JudgeKind, Verdict
from rag.generation.context import SourcesContextAssembler


def rebuild_sources(chunk_ids: Sequence[str], chunks: Mapping[str, Chunk]) -> str:
    """The ``<sources>`` block exactly as the generator's prompt showed it."""
    shown = [
        ScoredChunk(chunk=chunks[cid], score=0.0, rank=rank, stage="prompt")
        for rank, cid in enumerate(chunk_ids, start=1)
        if cid in chunks
    ]
    if not shown:
        return ""
    prompt = SourcesContextAssembler(max_sources=len(shown)).assemble(Query(text="-"), shown)
    return prompt.user.split("<sources>\n", 1)[1].split("\n</sources>", 1)[0]


class JudgedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    evidence_reach: str | None
    outcome: str
    numeric_match: bool | None
    verdicts: dict[str, Verdict] = Field(default_factory=dict)
    """Judge kind → verdict; a kind is missing if that judge failed."""


class JudgeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    judged: int
    skipped: dict[str, int]
    """Not judged, by reason: ``error`` (no usable answer), ``refused``."""
    rates: dict[str, float]
    """Kind → share of judged answers that passed (correct / faithful)."""
    by_reach: dict[str, dict[str, float]]
    """Evidence reach → kind → pass rate (the ladder, now for text answers too)."""
    numeric_agreement: dict[str, float]
    """On numeric questions: how often the correctness judge agrees with numeric_match."""
    failures: dict[str, int]


class JudgeRunner:
    def __init__(self, judges: Sequence[Judge], chunks: Mapping[str, Chunk]) -> None:
        if not judges:
            raise ValueError("at least one judge is required")
        self._judges = list(judges)
        self._chunks = chunks

    def case(self, trace: GenerationTrace) -> JudgeCase:
        return JudgeCase(
            item_id=trace.item_id,
            question=trace.question,
            reference_answer=trace.reference_answer,
            answer=trace.answer,
            working=trace.working,
            sources=rebuild_sources(trace.prompt_chunk_ids, self._chunks),
        )

    def run(self, traces: Sequence[GenerationTrace]) -> tuple[list[JudgedAnswer], JudgeSummary]:
        judged: list[JudgedAnswer] = []
        skipped: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        for trace in traces:
            if trace.outcome is AnswerOutcome.ERROR:
                skipped["error"] += 1
                continue
            if trace.refused:
                skipped["refused"] += 1
                continue
            case = self.case(trace)
            verdicts: dict[str, Verdict] = {}
            for judge in self._judges:
                verdict = judge.judge(case)
                if verdict is None:
                    failures[str(judge.kind)] += 1
                else:
                    verdicts[str(judge.kind)] = verdict
            judged.append(
                JudgedAnswer(
                    item_id=trace.item_id,
                    evidence_reach=str(trace.evidence_reach) if trace.evidence_reach else None,
                    outcome=str(trace.outcome),
                    numeric_match=trace.numeric_match,
                    verdicts=verdicts,
                )
            )
        return judged, _summarise(judged, skipped, failures)


def _rate(answers: Sequence[JudgedAnswer], kind: str) -> float:
    scored = [a.verdicts[kind].passed for a in answers if kind in a.verdicts]
    return sum(scored) / len(scored) if scored else 0.0


def _summarise(
    judged: Sequence[JudgedAnswer], skipped: Counter[str], failures: Counter[str]
) -> JudgeSummary:
    kinds = sorted({k for a in judged for k in a.verdicts})
    reaches = sorted({a.evidence_reach or "unknown" for a in judged})
    numeric = [
        a
        for a in judged
        if a.numeric_match is not None and str(JudgeKind.CORRECTNESS) in a.verdicts
    ]
    agree = [a.verdicts[str(JudgeKind.CORRECTNESS)].passed == a.numeric_match for a in numeric]
    return JudgeSummary(
        judged=len(judged),
        skipped=dict(skipped),
        rates={k: _rate(judged, k) for k in kinds},
        by_reach={
            r: {
                "n": float(
                    len(group := [a for a in judged if (a.evidence_reach or "unknown") == r])
                ),
                **{k: _rate(group, k) for k in kinds},
            }
            for r in reaches
        },
        numeric_agreement={
            "n": float(len(numeric)),
            "agreement": sum(agree) / len(agree) if agree else 0.0,
        },
        failures=dict(failures),
    )
