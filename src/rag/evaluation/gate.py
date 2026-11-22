"""Evaluation regression gate: compare a small golden subset with committed expectations
(study guide Part 13 / 14: "CI evaluation"; backlog 4.3; HLD 024).

The gate set is a few dev companies, chosen so that a company-filtered question searches
exactly the same filings in the small gate corpus as in the full corpus. Its per-question
recall@5 must then reproduce the full run, and any drop is a real regression, not a
sampling artefact. Test questions are never used: a check that runs on every PR would
leak the held-out split.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from rag.errors import ConfigError

EPSILON = 1e-9


class GateSpec(BaseModel):
    """``evals/gate-v1.json``: what to download, which questions, what to expect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    config: str
    """The retrieval config the gate checks (no LLM: it must run on a CI runner)."""
    split: str = "dev"
    companies: tuple[str, ...]
    doc_names: tuple[str, ...]
    expected: dict[str, float] = Field(default_factory=dict)
    """question ID → expected recall@5. Written by ``eval_gate.py --update``."""
    max_regressions: int = 1
    """Questions allowed to lose recall (near-tie rankings can flip across CPU/GPU)."""
    max_mean_drop: float = 0.05


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    mean_expected: float
    mean_actual: float
    regressions: dict[str, tuple[float, float]]
    """question ID → (expected, actual), for every question whose recall fell."""
    improvements: dict[str, tuple[float, float]]
    missing: tuple[str, ...]
    """Expected questions that were not evaluated (a broken gate is a failure)."""
    reasons: tuple[str, ...]


def check_gate(
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    max_regressions: int = 1,
    max_mean_drop: float = 0.05,
) -> GateResult:
    """Pass unless more than ``max_regressions`` questions lose recall, the mean recall
    drops by more than ``max_mean_drop``, or an expected question is missing. Gains never
    fail the gate; they are reported so the expectations can be updated deliberately."""
    if not expected:
        raise ConfigError("the gate has no expectations: run eval_gate.py --update first")
    missing = tuple(sorted(set(expected) - set(actual)))
    common = sorted(set(expected) & set(actual))
    regressions = {q: (expected[q], actual[q]) for q in common if actual[q] < expected[q] - EPSILON}
    improvements = {
        q: (expected[q], actual[q]) for q in common if actual[q] > expected[q] + EPSILON
    }
    mean_expected = fmean(expected.values())
    mean_actual = fmean(actual[q] for q in common) if common else 0.0
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} expected questions were not evaluated")
    if len(regressions) > max_regressions:
        reasons.append(f"{len(regressions)} questions lost recall (allowed {max_regressions})")
    if mean_expected - mean_actual > max_mean_drop + EPSILON:
        reasons.append(
            f"mean recall@5 fell {mean_expected - mean_actual:.3f} (allowed {max_mean_drop})"
        )
    return GateResult(
        passed=not reasons,
        mean_expected=mean_expected,
        mean_actual=mean_actual,
        regressions=regressions,
        improvements=improvements,
        missing=missing,
        reasons=tuple(reasons),
    )


def load_gate(path: Path) -> GateSpec:
    return GateSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
