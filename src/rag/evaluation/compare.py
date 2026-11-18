"""Is a difference between two runs real, or noise? (study guide, Part 13: ablation ladder)

With 150 questions, a 0.02 change in recall@5 is about three questions. A **paired
bootstrap** answers "how much could the mean difference move if we had drawn a different
set of questions?": resample the per-question differences with replacement many times and
read the 2.5th and 97.5th percentiles. If that interval contains 0, the runs are tied.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from statistics import fmean

from pydantic import BaseModel, ConfigDict


class RunComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    """mean(b - a): positive means run B is better."""
    ci_low: float
    ci_high: float
    better: int
    worse: int
    same: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def paired_bootstrap_ci(
    diffs: Sequence[float], resamples: int = 5000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean of paired differences."""
    if not diffs:
        raise ValueError("no differences to resample")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be >= 1 and confidence within (0, 1)")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(rng.choices(diffs, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(int((1 - tail) * resamples), resamples - 1)]
    return low, high


def compare_runs(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    metric: str,
    resamples: int = 5000,
    seed: int = 0,
) -> RunComparison:
    """Compare one metric between two runs over the questions they share (by item id)."""
    shared = sorted(scores_a.keys() & scores_b.keys())
    if not shared:
        raise ValueError("the runs share no questions")
    a = [scores_a[i] for i in shared]
    b = [scores_b[i] for i in shared]
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    low, high = paired_bootstrap_ci(diffs, resamples=resamples, seed=seed)
    return RunComparison(
        metric=metric,
        n=len(shared),
        mean_a=fmean(a),
        mean_b=fmean(b),
        mean_diff=fmean(diffs),
        ci_low=low,
        ci_high=high,
        better=sum(d > 0 for d in diffs),
        worse=sum(d < 0 for d in diffs),
        same=sum(d == 0 for d in diffs),
    )
