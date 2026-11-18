import pytest

from rag.observability import StageTimer


class FakeClock:
    def __init__(self, *ticks: float) -> None:
        self._ticks = list(ticks)

    def __call__(self) -> float:
        return self._ticks.pop(0)


def test_records_stage_duration_in_ms() -> None:
    timer = StageTimer(clock=FakeClock(1.0, 1.25))
    with timer.stage("dense"):
        pass
    assert timer.timings_ms == {"dense": 250.0}


def test_repeated_stage_accumulates_and_total_sums() -> None:
    timer = StageTimer(clock=FakeClock(0.0, 0.1, 0.1, 0.3, 0.3, 0.4))
    for name in ("dense", "dense", "rerank"):
        with timer.stage(name):
            pass
    assert timer.timings_ms["dense"] == pytest.approx(300.0)
    assert timer.total_ms == pytest.approx(400.0)


def test_records_duration_even_when_stage_raises() -> None:
    timer = StageTimer(clock=FakeClock(0.0, 0.5))
    with pytest.raises(RuntimeError), timer.stage("llm"):
        raise RuntimeError("boom")
    assert timer.timings_ms == {"llm": 500.0}


def test_timings_view_is_read_only() -> None:
    timer = StageTimer()
    with pytest.raises(TypeError):
        timer.timings_ms["x"] = 1.0  # type: ignore[index]
