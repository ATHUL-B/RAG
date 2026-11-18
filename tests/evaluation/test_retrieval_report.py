import pytest

from rag.evaluation.retrieval_report import (
    OVERALL,
    QueryMetrics,
    RetrievalEvaluator,
    aggregate,
)

QRELS = {"A": 1, "B": 1}


class TestRetrievalEvaluator:
    def test_scores_every_metric_at_every_k(self) -> None:
        result = RetrievalEvaluator(ks=[2, 1]).score("q1", ["A", "Z", "B"], QRELS, tags=["t"])
        assert result.item_id == "q1"
        assert result.tags == {"t"}
        assert result.scores == pytest.approx(
            {
                "mrr": 1.0,
                "recall@1": 0.5,
                "precision@1": 1.0,
                "hit@1": 1.0,
                "ndcg@1": 1.0,
                "recall@2": 0.5,
                "precision@2": 0.5,
                "hit@2": 1.0,
                "ndcg@2": 1 / (1 + 1 / 1.5849625007211562),
            }
        )

    def test_ks_are_sorted_and_deduplicated(self) -> None:
        assert RetrievalEvaluator(ks=[10, 1, 5, 5]).ks == (1, 5, 10)

    @pytest.mark.parametrize("ks", [[], [0, 5]])
    def test_rejects_invalid_ks(self, ks: list[int]) -> None:
        with pytest.raises(ValueError, match="ks"):
            RetrievalEvaluator(ks=ks)

    def test_unanswerable_query_raises(self) -> None:
        with pytest.raises(ValueError, match="unanswerable"):
            RetrievalEvaluator(ks=[1]).score("q", ["A"], {})


class TestAggregate:
    def test_reports_overall_and_per_tag_means_with_counts(self) -> None:
        results = [
            QueryMetrics(item_id="q1", tags=frozenset({"table"}), scores={"recall@5": 1.0}),
            QueryMetrics(
                item_id="q2", tags=frozenset({"table", "multi"}), scores={"recall@5": 0.0}
            ),
            QueryMetrics(item_id="q3", tags=frozenset(), scores={"recall@5": 0.5}),
        ]
        report = aggregate(results)
        assert report.overall.count == 3
        assert report.overall.means == {"recall@5": pytest.approx(0.5)}
        assert report.segments["table"].count == 2
        assert report.segments["table"].means == {"recall@5": pytest.approx(0.5)}
        assert report.segments["multi"].means == {"recall@5": 0.0}
        assert set(report.segments) == {OVERALL, "table", "multi"}

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="zero results"):
            aggregate([])

    def test_rejects_mixed_metric_sets(self) -> None:
        results = [
            QueryMetrics(item_id="q1", scores={"recall@5": 1.0}),
            QueryMetrics(item_id="q2", scores={"recall@10": 1.0}),
        ]
        with pytest.raises(ValueError, match="different metrics"):
            aggregate(results)
