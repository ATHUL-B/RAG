"""Run golden questions through a retriever; score, trace and localize failures (HLD 005)."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import Chunk, Filters, GoldenItem, Metadata, Query, ScoredChunk, matches
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
from rag.retrieval.base import RetrievalResult, Retriever

RETRIEVE_STAGE = "retrieve"


class RetrievalStatus(StrEnum):
    HIT = "hit"
    PARTIAL = "partial"
    NOT_RETRIEVED = "not_retrieved"
    FILTERED_OUT = "filtered_out"
    """The applied metadata filters excluded every evidence document."""
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
    applied_filters: Filters
    relaxed: list[str]
    queries: list[str] = Field(default_factory=list)
    """Search texts used when the query was rewritten (empty = the question as-is)."""
    stage_ranks: dict[str, int | None]
    """Per intermediate stage (e.g. dense, bm25): rank of the first evidence chunk, or None."""
    retrieved: list[RetrievedRecord]
    timings_ms: dict[str, float]
    metrics: dict[str, float]
    degraded: list[str] = Field(default_factory=list)
    """Stages that fell back or were cut short for this question (e.g. a rerank budget)."""


class RetrievalRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int
    traces: list[QueryTrace]
    report: MetricsReport
    status_counts: dict[RetrievalStatus, int]
    latency_ms: dict[str, float]
    """End-to-end retrieval latency percentiles."""
    stage_latency_ms: dict[str, dict[str, float]]
    """Percentiles per internal stage (e.g. first_stage, rerank), when reported."""
    skipped_unanswerable: list[str]


def doc_metadata_from_chunks(chunks: Iterable[Chunk]) -> dict[str, Metadata]:
    """Document-level metadata (company, year, ...) as copied onto each document's chunks."""
    metadata: dict[str, Metadata] = {}
    for chunk in chunks:
        metadata.setdefault(chunk.doc_id, chunk.metadata)
    return metadata


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
        doc_metadata: Mapping[str, Metadata] | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._retriever = retriever
        self._evaluator = evaluator
        self._indexed = indexed
        self._top_k = top_k
        self._doc_metadata = doc_metadata or {}
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
            stage_latency_ms=_stage_latency(traces),
            skipped_unanswerable=skipped,
        )

    def _run_one(self, item: GoldenItem) -> tuple[QueryTrace, QueryMetrics]:
        timer = StageTimer(self._clock)
        with timer.stage(RETRIEVE_STAGE):
            result = self._retriever.retrieve(Query(text=item.question), self._depth)
        hits = result.candidates

        qrels = qrels_for(item)
        ids = relevance_ids([h.chunk for h in hits], item)
        metrics = self._evaluator.score(item.item_id, ids, qrels, tags=item.tags)
        evidence_docs = {span.doc_id for span in item.evidence}
        doc_hit = any(h.chunk.doc_id in evidence_docs for h in hits[: self._top_k])
        filter_ok = self._filters_allow_evidence(result.applied_filters, evidence_docs)
        stage_ranks = {
            name: _first_relevant_rank(relevance_ids([c.chunk for c in stage], item), qrels)
            for name, stage in result.stage_candidates.items()
        }
        scores = {
            **metrics.scores,
            f"doc_hit@{self._top_k}": float(doc_hit),
            "filter_ok": float(filter_ok),
            "pool_recall": _pool_recall(result, item, qrels),
        }

        trace = QueryTrace(
            item_id=item.item_id,
            question=item.question,
            tags=sorted(item.tags),
            evidence_units=sorted(qrels),
            status=self._status(set(qrels), set(ids[: self._top_k]), filter_ok),
            applied_filters=dict(result.applied_filters),
            relaxed=list(result.relaxed),
            queries=list(result.queries),
            stage_ranks=stage_ranks,
            retrieved=_records(hits, ids, qrels.keys()),
            timings_ms={**timer.timings_ms, **result.timings_ms},
            metrics=scores,
            degraded=list(result.degraded),
        )
        return trace, metrics.model_copy(update={"scores": scores})

    def _filters_allow_evidence(self, filters: Filters, evidence_docs: set[str]) -> bool:
        """Would the applied filters let *any* evidence document through?

        Documents without known metadata are assumed allowed (we cannot tell).
        """
        if not filters:
            return True
        return any(
            doc not in self._doc_metadata or matches(self._doc_metadata[doc], filters)
            for doc in evidence_docs
        )

    def _status(self, evidence: set[str], found: set[str], filter_ok: bool) -> RetrievalStatus:
        if not evidence & set(self._indexed):
            return RetrievalStatus.NOT_IN_INDEX
        hits = evidence & found
        if not hits and not filter_ok:
            return RetrievalStatus.FILTERED_OUT
        if not hits:
            return RetrievalStatus.NOT_RETRIEVED
        return RetrievalStatus.HIT if hits == evidence else RetrievalStatus.PARTIAL


def _stage_latency(traces: Sequence[QueryTrace]) -> dict[str, dict[str, float]]:
    stages = sorted({name for t in traces for name in t.timings_ms} - {RETRIEVE_STAGE})
    return {
        name: latency_summary([t.timings_ms[name] for t in traces if name in t.timings_ms])
        for name in stages
    }


def _first_relevant_rank(ids: Sequence[str], qrels: Collection[str]) -> int | None:
    return next((rank for rank, unit in enumerate(ids, start=1) if unit in qrels), None)


def _pool_recall(result: RetrievalResult, item: GoldenItem, qrels: Collection[str]) -> float:
    """Share of evidence units anywhere in the candidate pool (all stages + final list).

    Bounds what a reranker could achieve: it can only reorder what it is given (Part 13).
    """
    pool = [c.chunk for c in result.candidates]
    pool += [c.chunk for stage in result.stage_candidates.values() for c in stage]
    found = set(relevance_ids(pool, item)) & set(qrels)
    return len(found) / len(qrels)


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
