import pytest

from rag.domain import Chunk, Filters, GoldenItem, Metadata, Query, ScoredChunk, SourceSpan
from rag.evaluation.retrieval_report import RetrievalEvaluator
from rag.evaluation.retrieval_runner import (
    RetrievalEvalRunner,
    RetrievalStatus,
    doc_metadata_from_chunks,
    indexed_units,
)
from rag.retrieval.base import RetrievalResult, Retriever


def _chunk(chunk_id: str, doc_id: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="t",
        char_start=0,
        char_end=1,
        page_start=page,
        page_end=page,
    )


class ScriptedRetriever(Retriever):
    """Returns a fixed chunk list per question text."""

    def __init__(
        self,
        results: dict[str, list[Chunk]],
        filters: Filters | None = None,
        relaxed: tuple[str, ...] = (),
    ) -> None:
        self._results = results
        self._filters = filters or {}
        self._relaxed = relaxed
        self.depths: list[int] = []

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.depths.append(k)
        chunks = self._results[query.text][:k]
        hits = tuple(
            ScoredChunk(chunk=c, score=1.0 - i / 10, rank=i + 1, stage="dense")
            for i, c in enumerate(chunks)
        )
        return RetrievalResult(
            candidates=hits, applied_filters=self._filters, relaxed=self._relaxed
        )


class TickClock:
    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _item(
    item_id: str, *pages: int, doc: str = "D", tags: frozenset[str] = frozenset()
) -> GoldenItem:
    return GoldenItem(
        item_id=item_id,
        question=item_id,
        reference_answer="a",
        evidence=tuple(SourceSpan(doc_id=doc, page_index=p) for p in pages),
        tags=tags,
    )


NOISE = [_chunk(f"n{i}", "OTHER", i) for i in range(5)]
RESULTS = {
    "hit": [_chunk("h", "D", 1), *NOISE],
    "partial": [_chunk("p", "D", 1), *NOISE],
    "miss_right_doc": [*NOISE[:2], _chunk("w", "D", 7), *NOISE[2:]],
    "deep": [*NOISE, _chunk("late", "D", 1)],
    "unindexed": NOISE,
}
INDEXED = indexed_units([_chunk("x", "D", 1), _chunk("y", "D", 2), _chunk("z", "D", 7)])


def _runner(retriever: Retriever, top_k: int = 3) -> RetrievalEvalRunner:
    return RetrievalEvalRunner(
        retriever, RetrievalEvaluator(ks=[1, 5]), INDEXED, top_k=top_k, clock=TickClock(0.01)
    )


def test_indexed_units_cover_every_page_of_multi_page_chunks() -> None:
    chunk = _chunk("c", "D", 3).model_copy(update={"page_end": 4})
    assert indexed_units([chunk]) == {"D#p3", "D#p4"}


def test_classifies_each_question_on_the_failure_ladder() -> None:
    items = [
        _item("hit", 1),
        _item("partial", 1, 2),
        _item("miss_right_doc", 1),
        _item("deep", 1),
        _item("unindexed", 99),
    ]
    result = _runner(ScriptedRetriever(RESULTS)).run(items)
    status = {t.item_id: t.status for t in result.traces}

    assert status == {
        "hit": RetrievalStatus.HIT,
        "partial": RetrievalStatus.PARTIAL,
        "miss_right_doc": RetrievalStatus.NOT_RETRIEVED,
        "deep": RetrievalStatus.NOT_RETRIEVED,  # found, but below top_k=3
        "unindexed": RetrievalStatus.NOT_IN_INDEX,
    }
    assert result.status_counts[RetrievalStatus.NOT_RETRIEVED] == 2


def test_retrieves_deep_enough_for_every_metric_cutoff() -> None:
    retriever = ScriptedRetriever(RESULTS)
    result = _runner(retriever, top_k=3).run([_item("deep", 1)])
    assert retriever.depths == [5]  # max(ks=[1, 5], top_k=3)
    (trace,) = result.traces
    assert trace.metrics["recall@5"] == 0.0  # 'late' is 6th: beyond depth 5
    assert len(trace.retrieved) == 5


def test_doc_hit_separates_wrong_document_from_wrong_page() -> None:
    result = _runner(ScriptedRetriever(RESULTS)).run(
        [_item("miss_right_doc", 1), _item("unindexed", 99)]
    )
    doc_hit = {t.item_id: t.metrics["doc_hit@3"] for t in result.traces}
    assert doc_hit == {"miss_right_doc": 1.0, "unindexed": 0.0}
    assert result.report.overall.means["doc_hit@3"] == 0.5


def test_trace_records_ranked_candidates_with_relevance() -> None:
    result = _runner(ScriptedRetriever(RESULTS)).run([_item("hit", 1, tags=frozenset({"t"}))])
    (trace,) = result.traces
    assert trace.evidence_units == ["D#p1"]
    assert trace.tags == ["t"]
    first = trace.retrieved[0]
    assert (first.rank, first.chunk_id, first.relevance_id, first.relevant) == (
        1,
        "h",
        "D#p1",
        True,
    )
    assert trace.retrieved[1].relevant is False
    assert trace.timings_ms == {"retrieve": pytest.approx(10.0)}
    assert result.latency_ms["p50"] == pytest.approx(10.0)
    assert result.report.segments["t"].count == 1


def test_unanswerable_items_are_skipped_and_listed() -> None:
    unanswerable = GoldenItem(item_id="u", question="u", reference_answer="", evidence=())
    result = _runner(ScriptedRetriever(RESULTS)).run([unanswerable, _item("hit", 1)])
    assert result.skipped_unanswerable == ["u"]
    assert [t.item_id for t in result.traces] == ["hit"]


def test_rejects_empty_runs_and_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="no answerable"):
        _runner(ScriptedRetriever(RESULTS)).run([])
    with pytest.raises(ValueError, match="top_k"):
        RetrievalEvalRunner(ScriptedRetriever({}), RetrievalEvaluator([1]), set(), top_k=0)


DOC_METADATA: dict[str, Metadata] = {
    "D": {"company": "ACME", "year": 2018},
    "OTHER": {"company": "ACME", "year": 2022},
}


def _filtered_runner(filters: Filters, relaxed: tuple[str, ...] = ()) -> RetrievalEvalRunner:
    return RetrievalEvalRunner(
        ScriptedRetriever(RESULTS, filters, relaxed),
        RetrievalEvaluator(ks=[5]),
        INDEXED,
        top_k=3,
        doc_metadata=DOC_METADATA,
        clock=TickClock(0.01),
    )


def test_filter_that_excludes_the_evidence_is_its_own_failure() -> None:
    result = _filtered_runner({"company": "ACME", "year": 2022}).run([_item("unindexed", 1)])
    (trace,) = result.traces
    assert trace.status is RetrievalStatus.FILTERED_OUT
    assert trace.metrics["filter_ok"] == 0.0
    assert trace.applied_filters == {"company": "ACME", "year": 2022}


def test_correct_filter_is_recorded_with_relaxation() -> None:
    result = _filtered_runner({"company": "ACME", "year": (2017, 2018)}, relaxed=("doc_type",)).run(
        [_item("hit", 1)]
    )
    (trace,) = result.traces
    assert trace.status is RetrievalStatus.HIT
    assert trace.metrics["filter_ok"] == 1.0
    assert trace.relaxed == ["doc_type"]


def test_unfiltered_runs_count_as_filter_ok() -> None:
    (trace,) = _runner(ScriptedRetriever(RESULTS)).run([_item("hit", 1)]).traces
    assert trace.metrics["filter_ok"] == 1.0
    assert trace.applied_filters == {}


def test_doc_metadata_comes_from_the_first_chunk_of_each_document() -> None:
    a = _chunk("a", "D", 1).model_copy(update={"metadata": {"year": 2018}})
    b = _chunk("b", "D", 2).model_copy(update={"metadata": {"year": 1999}})
    assert doc_metadata_from_chunks([a, b]) == {"D": {"year": 2018}}


class StagedRetriever(Retriever):
    """Final list misses the evidence, but the bm25 stage found it at rank 2."""

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        noise = tuple(
            ScoredChunk(chunk=c, score=0.5, rank=i, stage="rrf") for i, c in enumerate(NOISE, 1)
        )
        bm25 = (noise[0], ScoredChunk(chunk=_chunk("e", "D", 1), score=1.0, rank=2, stage="bm25"))
        return RetrievalResult(
            candidates=noise[:k], stage_candidates={"dense": noise, "bm25": bm25}
        )


def test_stage_ranks_and_pool_recall_show_where_evidence_was_lost() -> None:
    (trace,) = _runner(StagedRetriever()).run([_item("x", 1, 2)]).traces
    assert trace.stage_ranks == {"dense": None, "bm25": 2}
    assert trace.metrics["pool_recall"] == 0.5  # page 1 in the pool, page 2 nowhere
    assert trace.metrics["recall@5"] == 0.0  # ...but fusion dropped it from the final list


def test_pool_recall_without_stages_is_recall_over_the_returned_list() -> None:
    (trace,) = _runner(ScriptedRetriever(RESULTS)).run([_item("deep", 1)]).traces
    assert trace.stage_ranks == {}
    assert trace.metrics["pool_recall"] == 0.0


class TimedRetriever(Retriever):
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            candidates=(), timings_ms={"first_stage": 10.0 * self.calls, "rerank": 100.0}
        )


def test_internal_stage_timings_are_traced_and_summarized() -> None:
    result = _runner(TimedRetriever()).run([_item("a", 1), _item("b", 1)])
    assert result.traces[0].timings_ms["rerank"] == 100.0
    assert set(result.stage_latency_ms) == {"first_stage", "rerank"}
    assert result.stage_latency_ms["first_stage"]["max"] == 20.0
    assert result.stage_latency_ms["rerank"]["p50"] == 100.0
