import pytest

from rag.evaluation.calibration import agreement, cohens_kappa


def test_kappa_hand_computed() -> None:
    human = ["c", "c", "c", "c", "i", "i"]
    judge = ["c", "c", "c", "i", "i", "c"]
    # observed 4/6; chance (4*4 + 2*2) / 36 = 20/36; kappa = (24/36 - 20/36) / (16/36) = 0.25
    assert cohens_kappa(human, judge) == pytest.approx(0.25)


def test_always_saying_correct_agrees_80_percent_but_kappa_is_zero() -> None:
    human = ["c"] * 8 + ["i"] * 2
    judge = ["c"] * 10
    report = agreement(human, judge)
    assert report.observed == pytest.approx(0.8)
    assert report.kappa == pytest.approx(0.0)  # it learned nothing beyond the base rate
    assert report.confusion == {"c→c": 8, "i→c": 2}


def test_perfect_and_degenerate_agreement() -> None:
    assert cohens_kappa(["c", "i", "i"], ["c", "i", "i"]) == pytest.approx(1.0)
    assert cohens_kappa(["c", "c"], ["c", "c"]) == 1.0  # one shared label, full agreement


def test_rejects_mismatched_or_empty_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(["c"], ["c", "i"])
    with pytest.raises(ValueError, match="no labels"):
        cohens_kappa([], [])
