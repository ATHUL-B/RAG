"""Latency summaries: always report tails (p95/p99), not just averages (Part 14)."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import quantiles


def latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / p99 / max of a sample; empty dict for an empty sample."""
    if not values_ms:
        return {}
    if len(values_ms) == 1:
        (only,) = values_ms
        return {"p50": only, "p95": only, "p99": only, "max": only}
    cuts = quantiles(values_ms, n=100, method="inclusive")
    return {"p50": cuts[49], "p95": cuts[94], "p99": cuts[98], "max": max(values_ms)}
