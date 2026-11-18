import pytest

from rag.evaluation.compare import compare_runs, paired_bootstrap_ci


def test_identical_runs_are_a_tie_with_zero_interval() -> None:
    scores = {"q1": 1.0, "q2": 0.0, "q3": 0.5}
    result = compare_runs(scores, scores, "recall@5")
    assert (result.mean_diff, result.ci_low, result.ci_high) == (0.0, 0.0, 0.0)
    assert (result.better, result.worse, result.same) == (0, 0, 3)
    assert not result.significant


def test_consistent_improvement_is_significant() -> None:
    a = {f"q{i}": 0.0 for i in range(40)}
    b = {f"q{i}": 1.0 if i % 2 else 0.5 for i in range(40)}
    result = compare_runs(a, b, "recall@5")
    assert result.mean_diff == pytest.approx(0.75)
    assert result.ci_low > 0
    assert result.significant
    assert result.better == 40


def test_mixed_small_changes_are_not_significant() -> None:
    a = {f"q{i}": 0.0 for i in range(20)}
    b = {f"q{i}": (1.0 if i < 3 else -1.0 if i < 5 else 0.0) for i in range(20)}
    result = compare_runs(a, b, "m")
    assert (result.better, result.worse, result.same) == (3, 2, 15)
    assert result.ci_low < 0 < result.ci_high
    assert not result.significant


def test_only_shared_questions_are_compared() -> None:
    result = compare_runs({"q1": 0.0, "q2": 1.0}, {"q2": 1.0, "q3": 0.0}, "m")
    assert result.n == 1


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    diffs = [0.0, 1.0, -0.5, 0.25, 0.0, 1.0]
    assert paired_bootstrap_ci(diffs, seed=7) == paired_bootstrap_ci(diffs, seed=7)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="share no questions"):
        compare_runs({"a": 1.0}, {"b": 1.0}, "m")
    with pytest.raises(ValueError, match="no differences"):
        paired_bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_ci([1.0], confidence=1.5)
