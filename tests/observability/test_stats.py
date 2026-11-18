import pytest

from rag.observability.stats import latency_summary


def test_empty_sample_has_no_summary() -> None:
    assert latency_summary([]) == {}


def test_single_value_is_every_percentile() -> None:
    assert latency_summary([7.0]) == {"p50": 7.0, "p95": 7.0, "p99": 7.0, "max": 7.0}


def test_tail_percentiles_reflect_slow_requests() -> None:
    summary = latency_summary([float(v) for v in range(1, 101)])
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p95"] == pytest.approx(95.05)
    assert summary["max"] == 100.0
