"""Run golden questions through a retriever; score, trace and localize failures (HLD 005)."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from rag.domain import Chunk, GoldenItem, Query, ScoredChunk
from rag.evaluation.evidence import evidence_unit, qrels_for, relevance_ids
from rag.evaluation.retrieval_report import (
    MetricsReport,
    QueryMetrics,
    RetrievalEvaluator,
    aggregate,
)
from rag.observability import StageTimer
from rag.observability.stats import latency_summary
from rag.observability.timing import Clock
from rag.retrieval.base import Retriever

RETRIEVE_STAGE = "retrieve"


class RetrievalStatus(StrEnum):
    HIT = "hit"
    PARTIAL = "partial"
    NOT_RETRIEVED = "not_retrieved"
    NOT_IN_INDEX = "not_in_index"


class RetrievedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    score: float
    relevance_id: str
    relevant: bool


class QueryTrace(BaseModel):
    """Everything needed to debug one question without re-running it."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    question: str
    tags: list[str]
    evidence_units: list[str]
    status: RetrievalStatus
    retrieved: list[RetrievedRecord]
    timings_ms: dict[str, float]
    metrics: dict[str, float]


class RetrievalRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int
    traces: list[QueryTrace]
    report: MetricsReport
    status_counts: dict[RetrievalStatus, int]
    latency_ms: dict[str, float]
    skipped_unanswerable: list[str]


def indexed_units(chunks: Iterable[Chunk]) -> set[str]:
    """Every evidence unit (doc#page) that at least one indexed chunk covers."""
    return {
        evidence_unit(c.doc_id, page)
        for c in chunks
        for page in range(c.page_start, c.page_end + 1)
    }


class RetrievalEvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        evaluator: RetrievalEvaluator,
        indexed: Collection[str],
        top_k: int,
        clock: Clock = time.perf_counter,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._retriever = retriever
        self._evaluator = evaluator
        self._indexed = indexed
        self._top_k = top_k
        self._depth = max((*evaluator.ks, top_k))
        self._clock = clock

    def run(self, items: Iterable[GoldenItem]) -> RetrievalRunResult:
        traces: list[QueryTrace] = []
        results: list[QueryMetrics] = []
        skipped: list[str] = []
        for item in items:
            if not item.answerable:
                skipped.append(item.item_id)
                continue
            trace, metrics = self._run_one(item)
            traces.append(trace)
            results.append(metrics)
        if not traces:
            raise ValueError("no answerable items to evaluate")
        return RetrievalRunResult(
            top_k=self._top_k,
            traces=traces,
            report=aggregate(results),
            status_counts=dict(Counter(t.status for t in traces)),
            latency_ms=latency_summary([t.timings_ms[RETRIEVE_STAGE] for t in traces]),
            skipped_unanswerable=skipped,
        )

    def _run_one(self, item: GoldenItem) -> tuple[QueryTrace, QueryMetrics]:
        timer = StageTimer(self._clock)
        with timer.stage(RETRIEVE_STAGE):
            hits = self._retriever.retrieve(Query(text=item.question), self._depth)

        qrels = qrels_for(item)
        ids = relevance_ids([h.chunk for h in hits], item)
        metrics = self._evaluator.score(item.item_id, ids, qrels, tags=item.tags)
        evidence_docs = {span.doc_id for span in item.evidence}
        doc_hit = any(h.chunk.doc_id in evidence_docs for h in hits[: self._top_k])
        scores = {**metrics.scores, f"doc_hit@{self._top_k}": float(doc_hit)}

        trace = QueryTrace(
            item_id=item.item_id,
            question=item.question,
            tags=sorted(item.tags),
            evidence_units=sorted(qrels),
            status=self._status(set(qrels), set(ids[: self._top_k])),
            retrieved=_records(hits, ids, qrels.keys()),
            timings_ms=dict(timer.timings_ms),
            metrics=scores,
        )
        return trace, metrics.model_copy(update={"scores": scores})

    def _status(self, evidence: set[str], found: set[str]) -> RetrievalStatus:
        if not evidence & set(self._indexed):
            return RetrievalStatus.NOT_IN_INDEX
        hits = evidence & found
        if not hits:
            return RetrievalStatus.NOT_RETRIEVED
        return RetrievalStatus.HIT if hits == evidence else RetrievalStatus.PARTIAL


def _records(
    hits: Sequence[ScoredChunk], ids: Sequence[str], relevant: Collection[str]
) -> list[RetrievedRecord]:
    return [
        RetrievedRecord(
            rank=hit.rank,
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            page_start=hit.chunk.page_start,
            page_end=hit.chunk.page_end,
            score=hit.score,
            relevance_id=unit,
            relevant=unit in relevant,
        )
        for hit, unit in zip(hits, ids, strict=True)
    ]
