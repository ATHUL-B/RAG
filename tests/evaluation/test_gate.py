import pytest

from rag.errors import ConfigError
from rag.evaluation.gate import GateSpec, check_gate

EXPECTED = {"q1": 1.0, "q2": 0.5, "q3": 0.0, "q4": 1.0}


def test_identical_results_pass() -> None:
    result = check_gate(EXPECTED, dict(EXPECTED))
    assert result.passed and not result.regressions and not result.reasons
    assert result.mean_actual == result.mean_expected == pytest.approx(0.625)


def test_one_flip_is_tolerated_two_are_not() -> None:
    one = check_gate(EXPECTED, {**EXPECTED, "q1": 0.0}, max_regressions=1, max_mean_drop=1.0)
    assert one.passed and one.regressions == {"q1": (1.0, 0.0)}
    two = check_gate(
        EXPECTED, {**EXPECTED, "q1": 0.0, "q4": 0.0}, max_regressions=1, max_mean_drop=1.0
    )
    assert not two.passed and "2 questions lost recall" in two.reasons[0]


def test_mean_drop_fails_even_within_the_flip_allowance() -> None:
    # one question 1.0 -> 0.0: mean 0.625 -> 0.375, a 0.25 drop
    result = check_gate(EXPECTED, {**EXPECTED, "q1": 0.0}, max_regressions=1, max_mean_drop=0.05)
    assert not result.passed and "mean recall@5 fell 0.250" in result.reasons[0]


def test_gains_never_fail_and_are_reported() -> None:
    result = check_gate(EXPECTED, {**EXPECTED, "q3": 1.0})
    assert result.passed and result.improvements == {"q3": (0.0, 1.0)}


def test_missing_questions_fail() -> None:
    actual = {q: v for q, v in EXPECTED.items() if q != "q2"}
    result = check_gate(EXPECTED, actual)
    assert not result.passed and result.missing == ("q2",)


def test_no_expectations_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="--update"):
        check_gate({}, {"q1": 1.0})


def test_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        GateSpec.model_validate(
            {"version": 1, "config": "c", "companies": [], "doc_names": [], "typo": 1}
        )
