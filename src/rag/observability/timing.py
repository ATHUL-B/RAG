"""Per-stage wall-clock timing (study guide, Part 14: measure before optimizing).

The clock is injected so tests are deterministic and never sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType

Clock = Callable[[], float]


class StageTimer:
    """Accumulates elapsed milliseconds per named stage for one request.

    Example::

        timer = StageTimer()
        with timer.stage("dense_retrieval"):
            ...
        timer.timings_ms  # {"dense_retrieval": 12.3}
    """

    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the ``with`` block; repeated names accumulate. Recorded even on error."""
        start = self._clock()
        try:
            yield
        finally:
            elapsed_ms = (self._clock() - start) * 1000.0
            self._timings_ms[name] = self._timings_ms.get(name, 0.0) + elapsed_ms

    @property
    def timings_ms(self) -> Mapping[str, float]:
        return MappingProxyType(self._timings_ms)

    @property
    def total_ms(self) -> float:
        return sum(self._timings_ms.values())
