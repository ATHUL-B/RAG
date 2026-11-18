"""Retrieve → generate → check, per golden question, with cost accounting (HLD 010)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import GoldenItem, Query
from rag.errors import GenerationError
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


class GenerationRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    traces: list[GenerationTrace]
    report: MetricsReport | None
    numeric: dict[str, float]
    totals: dict[str, float]
    latency_ms: dict[str, dict[str, float]]
    errors: int


class GenerationEvalRunner:
    """One question at a time: retrieve ``depth`` candidates, generate, run the checks."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        pricing: Pricing,
        depth: int = 5,
        clock: Clock = time.perf_counter,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self._retriever = retriever
        self._generator = generator
        self._pricing = pricing
        self._depth = depth
        self._clock = clock

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
        try:
            with timer.stage(GENERATE_STAGE):
                result = self._generator.generate(query, candidates)
        except GenerationError as exc:
            logger.error("generation failed", extra={"item_id": item.item_id, "error": str(exc)})
            return GenerationTrace(**base, timings_ms=dict(timer.timings_ms), error=str(exc))

        answer, completion = result.answer, result.completion
        chunks = {c.chunk.chunk_id: c.chunk for c in candidates}
        cited = [chunks[cid] for cid in answer.cited_chunk_ids if cid in chunks]
        shown = [chunks[cid] for cid in result.prompt_chunk_ids if cid in chunks]
        qrels = qrels_for(item)
        reference = reference_number(item.reference_answer)
        answered = not answer.refused
        return GenerationTrace(
            **base,
            prompt_chunk_ids=list(result.prompt_chunk_ids),
            answer=answer.text,
            refused=answer.refused,
            provider_refused=completion.refused,
            model_id=completion.model_id,
            cited_chunk_ids=list(answer.cited_chunk_ids),
            invalid_citations=list(result.invalid_citations),
            numeric_match=_numeric_check(answer.text, reference, answered),
            metrics={
                "answered": float(answered),
                "evidence_in_prompt": float(bool(set(relevance_ids(shown, item)) & set(qrels))),
                "citation_valid": float(not result.invalid_citations),
                "has_citation": float(bool(answer.cited_chunk_ids)) if answered else 0.0,
                "cites_evidence": float(cites_evidence(cited, item)),
            },
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=self._pricing.cost(completion.input_tokens, completion.output_tokens),
            timings_ms=dict(timer.timings_ms),
        )


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
