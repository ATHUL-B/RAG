"""Hand-computed cases, many taken from the study guide's ZenithCare example (Part 13).

Relevant set for the hero question: {A, B, C}. D, E, F, G, K are not relevant.
"""

import math

import pytest

from rag.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

HERO_QRELS = {"A": 1, "B": 1, "C": 1}
NAIVE_TOP5 = ["F", "D", "E", "A", "G"]  # guide Part 6: only A found, at rank 4
FINAL_TOP4 = ["A", "B", "C", "K"]  # guide Part 10: K is supporting, not relevant


class TestGuideExamples:
    def test_naive_recall_at_5_is_one_third(self) -> None:
        assert recall_at_k(NAIVE_TOP5, HERO_QRELS, 5) == pytest.approx(1 / 3)

    def test_naive_precision_at_5_is_one_fifth(self) -> None:
        assert precision_at_k(NAIVE_TOP5, HERO_QRELS, 5) == pytest.approx(1 / 5)

    def test_naive_first_relevant_at_rank_4_gives_rr_quarter(self) -> None:
        assert reciprocal_rank(NAIVE_TOP5, HERO_QRELS) == pytest.approx(0.25)

    def test_final_recall_is_full_and_precision_three_quarters(self) -> None:
        assert recall_at_k(FINAL_TOP4, HERO_QRELS, 4) == 1.0
        assert precision_at_k(FINAL_TOP4, HERO_QRELS, 4) == pytest.approx(0.75)
        assert reciprocal_rank(FINAL_TOP4, HERO_QRELS) == 1.0

    def test_graded_ndcg_worked_example(self) -> None:
        # Top 4 graded [0, 3, 2, 0]; ideal order [3, 2] -> 2.893 / 4.262 = 0.679
        ranked = ["n1", "x", "y", "n2"]
        qrels = {"x": 3, "y": 2}
        expected = (3 / math.log2(3) + 2 / math.log2(4)) / (3 / 1 + 2 / math.log2(3))
        assert expected == pytest.approx(0.679, abs=1e-3)
        assert ndcg_at_k(ranked, qrels, 4) == pytest.approx(expected)

    def test_moving_perfect_chunk_to_rank_one_gives_ndcg_one(self) -> None:
        assert ndcg_at_k(["x", "y", "n1", "n2"], {"x": 3, "y": 2}, 4) == pytest.approx(1.0)


class TestCutoffs:
    def test_k_limits_what_counts(self) -> None:
        assert recall_at_k(NAIVE_TOP5, HERO_QRELS, 3) == 0.0
        assert hit_at_k(NAIVE_TOP5, HERO_QRELS, 3) == 0.0
        assert hit_at_k(NAIVE_TOP5, HERO_QRELS, 4) == 1.0

    def test_short_list_precision_still_divides_by_k(self) -> None:
        assert precision_at_k(["A"], HERO_QRELS, 5) == pytest.approx(0.2)

    def test_empty_ranking_scores_zero(self) -> None:
        assert recall_at_k([], HERO_QRELS, 5) == 0.0
        assert precision_at_k([], HERO_QRELS, 5) == 0.0
        assert reciprocal_rank([], HERO_QRELS) == 0.0
        assert ndcg_at_k([], HERO_QRELS, 5) == 0.0

    @pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_at_k, ndcg_at_k])
    def test_k_below_one_is_rejected(self, metric: object) -> None:
        with pytest.raises(ValueError, match="k must be"):
            metric(["A"], HERO_QRELS, 0)  # type: ignore[operator]


class TestDuplicates:
    """Two chunks from the same evidence page map to the same ID."""

    def test_recall_counts_a_repeated_item_once(self) -> None:
        assert recall_at_k(["A", "A", "A"], HERO_QRELS, 3) == pytest.approx(1 / 3)

    def test_precision_counts_every_relevant_position(self) -> None:
        assert precision_at_k(["A", "A", "Z"], HERO_QRELS, 3) == pytest.approx(2 / 3)

    def test_ndcg_never_exceeds_one_with_repeats(self) -> None:
        assert ndcg_at_k(["x", "x", "x"], {"x": 3}, 3) == pytest.approx(1.0)


class TestRelevanceGrades:
    def test_missing_relevant_items_lower_ndcg(self) -> None:
        # Only one of two equally relevant items retrieved, at rank 1.
        assert ndcg_at_k(["x"], {"x": 1, "y": 1}, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))

    def test_zero_or_negative_grades_are_not_relevant(self) -> None:
        qrels = {"A": 1, "N": 0, "M": -1}
        assert precision_at_k(["N", "M", "A"], qrels, 3) == pytest.approx(1 / 3)
        assert reciprocal_rank(["N", "M", "A"], qrels) == pytest.approx(1 / 3)
        assert ndcg_at_k(["M"], qrels, 1) == 0.0

    @pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
    def test_unanswerable_query_is_undefined(self, metric: object) -> None:
        with pytest.raises(ValueError, match="unanswerable"):
            metric(["A"], {"A": 0}, 1)  # type: ignore[operator]

    def test_unanswerable_query_has_zero_precision_hit_and_rr(self) -> None:
        assert precision_at_k(["A"], {}, 1) == 0.0
        assert hit_at_k(["A"], {}, 1) == 0.0
        assert reciprocal_rank(["A"], {}) == 0.0
