"""Retrieve → generate → check, per golden question, with cost accounting (HLD 010)."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from enum import StrEnum
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import GoldenItem, Query, ScoredChunk
from rag.errors import GenerationError, UnusableAnswerError
from rag.evaluation.evidence import qrels_for, relevance_ids
from rag.evaluation.generation_metrics import cites_evidence, numeric_match, reference_number
from rag.evaluation.retrieval_report import MetricsReport, QueryMetrics, aggregate
from rag.generation.base import AnswerGenerator, Prompt
from rag.ingestion.tokenization import TokenCounter
from rag.observability import StageTimer
from rag.observability.stats import latency_summary
from rag.observability.timing import Clock
from rag.retrieval.base import Retriever

logger = logging.getLogger(__name__)

RETRIEVE_STAGE = "retrieve"
GENERATE_STAGE = "generate"
API_ERROR = "api_error"


class EvidenceReach(StrEnum):
    """How far the labelled evidence got: the retrieval half of the debugging ladder (Part 13).

    Ordered from earliest loss to success; each value names the *first* stage that lost it.
    """

    NOT_IN_INDEX = "not_in_index"  # rung 1: ingestion (parse/chunk)
    NOT_RETRIEVED = "not_retrieved"  # rung 2: query/filter/embedding/hybrid
    NOT_IN_PROMPT = "not_in_prompt"  # rung 4: context assembly dropped it
    IN_PROMPT = "in_prompt"


class AnswerOutcome(StrEnum):
    """What the model did with the prompt: the generation half of the ladder (rung 5)."""

    ERROR = "error"  # no usable answer (API failure, schema break, empty/markup output)
    REFUSED = "refused"
    CORRECT = "correct"  # numeric reference matched
    WRONG = "wrong"  # numeric reference not matched
    UNSCORED = "unscored"  # answered, but the reference is text: needs a judge (2.2)


class Pricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0.0)
    output_per_mtok: float = Field(ge=0.0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class GenerationTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    question: str
    reference_answer: str
    tags: list[str]
    evidence_units: list[str]
    prompt_chunk_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    working: str = ""
    """Worked steps before the answer; never scored by numeric_match."""
    refused: bool = False
    provider_refused: bool = False
    model_id: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    numeric_match: bool | None = None
    """None when the reference answer is not a plain number."""
    metrics: dict[str, float] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    error_kind: str | None = None
    """``UnusableAnswerError.kind`` (e.g. "empty_answer"), or "api_error"."""
    raw_output: str = ""
    """The model's exact text before parsing, kept so failures can be diagnosed later."""
    evidence_reach: EvidenceReach | None = None
    outcome: AnswerOutcome = AnswerOutcome.ERROR


class GenerationRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    traces: list[GenerationTrace]
    report: MetricsReport | None
    numeric: dict[str, float]
    totals: dict[str, float]
    latency_ms: dict[str, dict[str, float]]
    errors: int
    error_kinds: dict[str, int] = Field(default_factory=dict)
    ladder: dict[str, dict[str, int]] = Field(default_factory=dict)
    """evidence_reach → outcome → count: where the evidence got to vs what the model did."""


class GenerationEvalRunner:
    """One question at a time: retrieve ``depth`` candidates, generate, run the checks."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        pricing: Pricing,
        depth: int = 5,
        clock: Clock = time.perf_counter,
        indexed: Collection[str] | None = None,
    ) -> None:
        """``indexed``: every evidence unit some indexed chunk covers (``indexed_units``).

        When given, evidence missing from the index is reported as ``not_in_index`` (rung 1)
        instead of ``not_retrieved``.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self._retriever = retriever
        self._generator = generator
        self._pricing = pricing
        self._depth = depth
        self._clock = clock
        self._indexed = set(indexed) if indexed is not None else None

    def run(self, items: Iterable[GoldenItem]) -> GenerationRunResult:
        traces = [self._run_one(item) for item in items]
        if not traces:
            raise ValueError("no items to evaluate")
        ok = [t for t in traces if t.error is None]
        report = (
            aggregate(
                [
                    QueryMetrics(item_id=t.item_id, tags=frozenset(t.tags), scores=t.metrics)
                    for t in ok
                ]
            )
            if ok
            else None
        )
        numeric = [t.numeric_match for t in ok if t.numeric_match is not None]
        return GenerationRunResult(
            traces=traces,
            report=report,
            numeric={
                "n": len(numeric),
                "matched": sum(numeric),
                "rate": fmean(numeric) if numeric else 0.0,
            },
            totals={
                "input_tokens": sum(t.input_tokens for t in traces),
                "output_tokens": sum(t.output_tokens for t in traces),
                "cost_usd": sum(t.cost_usd for t in traces),
            },
            latency_ms={
                stage: latency_summary(
                    [t.timings_ms[stage] for t in traces if stage in t.timings_ms]
                )
                for stage in (RETRIEVE_STAGE, GENERATE_STAGE)
            },
            errors=len(traces) - len(ok),
            error_kinds=dict(Counter(t.error_kind for t in traces if t.error_kind)),
            ladder=_ladder(traces),
        )

    def _run_one(self, item: GoldenItem) -> GenerationTrace:
        timer = StageTimer(self._clock)
        query = Query(text=item.question)
        base = {
            "item_id": item.item_id,
            "question": item.question,
            "reference_answer": item.reference_answer,
            "tags": sorted(item.tags),
            "evidence_units": sorted(qrels_for(item)),
        }
        with timer.stage(RETRIEVE_STAGE):
            candidates = self._retriever.retrieve(query, self._depth).candidates
        qrels = qrels_for(item)
        retrieved = set(relevance_ids([c.chunk for c in candidates], item)) & set(qrels)
        try:
            with timer.stage(GENERATE_STAGE):
                result = self._generator.generate(query, candidates)
        except GenerationError as exc:
            logger.error("generation failed", extra={"item_id": item.item_id, "error": str(exc)})
            if not isinstance(exc, UnusableAnswerError):  # API/network: no model output
                return GenerationTrace(
                    **base, timings_ms=dict(timer.timings_ms), error=str(exc), error_kind=API_ERROR
                )
            return GenerationTrace(
                **base,
                prompt_chunk_ids=list(exc.prompt_chunk_ids),
                timings_ms=dict(timer.timings_ms),
                error=str(exc),
                error_kind=exc.kind,
                raw_output=exc.raw_output,
                evidence_reach=self._reach(
                    qrels, retrieved, self._evidence_in(exc.prompt_chunk_ids, candidates, item)
                ),
            )

        answer, completion = result.answer, result.completion
        chunks = {c.chunk.chunk_id: c.chunk for c in candidates}
        cited = [chunks[cid] for cid in answer.cited_chunk_ids if cid in chunks]
        in_prompt = self._evidence_in(result.prompt_chunk_ids, candidates, item)
        reference = reference_number(item.reference_answer)
        answered = not answer.refused
        numeric = _numeric_check(answer.text, reference, answered)
        return GenerationTrace(
            **base,
            prompt_chunk_ids=list(result.prompt_chunk_ids),
            answer=answer.text,
            working=result.working,
            refused=answer.refused,
            provider_refused=completion.refused,
            model_id=completion.model_id,
            cited_chunk_ids=list(answer.cited_chunk_ids),
            invalid_citations=list(result.invalid_citations),
            numeric_match=numeric,
            metrics={
                "answered": float(answered),
                "evidence_in_prompt": float(bool(in_prompt)),
                "citation_valid": float(not result.invalid_citations),
                "has_citation": float(bool(answer.cited_chunk_ids)) if answered else 0.0,
                "cites_evidence": float(cites_evidence(cited, item)),
            },
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=self._pricing.cost(completion.input_tokens, completion.output_tokens),
            timings_ms=dict(timer.timings_ms),
            raw_output=completion.text,
            evidence_reach=self._reach(qrels, retrieved, in_prompt),
            outcome=_outcome(answered, numeric),
        )

    @staticmethod
    def _evidence_in(
        chunk_ids: Sequence[str], candidates: Sequence[ScoredChunk], item: GoldenItem
    ) -> set[str]:
        """Evidence units covered by the chunks with these IDs (the prompt's sources)."""
        chunks = {c.chunk.chunk_id: c.chunk for c in candidates}
        shown = [chunks[cid] for cid in chunk_ids if cid in chunks]
        return set(relevance_ids(shown, item)) & set(qrels_for(item))

    def _reach(
        self, qrels: Collection[str], retrieved: set[str], in_prompt: set[str]
    ) -> EvidenceReach:
        """The first ladder stage that lost all the evidence (any evidence page counts)."""
        if self._indexed is not None and not set(qrels) & self._indexed:
            return EvidenceReach.NOT_IN_INDEX
        if not retrieved:
            return EvidenceReach.NOT_RETRIEVED
        if not in_prompt:
            return EvidenceReach.NOT_IN_PROMPT
        return EvidenceReach.IN_PROMPT


def _outcome(answered: bool, numeric: bool | None) -> AnswerOutcome:
    if not answered:
        return AnswerOutcome.REFUSED
    if numeric is None:
        return AnswerOutcome.UNSCORED
    return AnswerOutcome.CORRECT if numeric else AnswerOutcome.WRONG


def _ladder(traces: Sequence[GenerationTrace]) -> dict[str, dict[str, int]]:
    """Cross-tab of evidence reach x outcome, in ladder order (unknown reach last)."""
    table: dict[str, dict[str, int]] = {}
    order = [*EvidenceReach, None]
    for reach in order:
        rows = [t for t in traces if t.evidence_reach == reach]
        if rows:
            table[str(reach or "unknown")] = dict(Counter(str(t.outcome) for t in rows))
    return table


def _numeric_check(answer: str, reference: float | None, answered: bool) -> bool | None:
    """None when not applicable; a refusal on a numeric question counts as a miss."""
    if reference is None:
        return None
    return answered and numeric_match(answer, reference)


def estimate_cost(
    prompts: Sequence[Prompt],
    counter: TokenCounter,
    pricing: Pricing,
    expected_output_tokens: int,
    safety_margin: float = 1.2,
) -> float:
    """Rough upper estimate of a run's cost *before* calling a paid API.

    Prompt tokens are counted locally (an approximation of the provider's tokenizer, hence
    the margin); output tokens are a per-question expectation that includes thinking.
    """
    input_tokens = sum(counter.count(p.system) + counter.count(p.user) for p in prompts)
    output_tokens = expected_output_tokens * len(prompts)
    return pricing.cost(int(input_tokens * safety_margin), output_tokens)
