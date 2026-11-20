"""Judge calibration: how well does an LLM judge agree with a human? (Part 13; HLD 016).

Raw agreement is misleading when one label dominates: if 80% of answers are correct, a
judge that always says "correct" agrees 80% of the time and has learned nothing. Cohen's
kappa corrects for the agreement expected by chance:

    kappa = (p_observed - p_chance) / (1 - p_chance)

where p_chance = Σ_label P(human = label) · P(judge = label). 1 = perfect, 0 = chance level.
Common reading: > 0.6 substantial, > 0.8 almost perfect. The acceptance bar is fixed
*before* looking at the results.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Agreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    observed: float
    kappa: float
    confusion: dict[str, int]
    """``"<human>→<judge>"`` → count, e.g. ``"correct→incorrect": 3``."""


def cohens_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    if len(human) != len(judge):
        raise ValueError("human and judge labels must have the same length")
    if not human:
        raise ValueError("no labels to compare")
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    h_counts, j_counts = Counter(human), Counter(judge)
    chance = sum(h_counts[label] * j_counts[label] for label in h_counts) / (n * n)
    if chance == 1.0:  # both raters used a single, identical label: no information
        return 1.0 if observed == 1.0 else 0.0
    return (observed - chance) / (1.0 - chance)


def agreement(human: Sequence[str], judge: Sequence[str]) -> Agreement:
    kappa = cohens_kappa(human, judge)
    n = len(human)
    return Agreement(
        n=n,
        observed=sum(h == j for h, j in zip(human, judge, strict=True)) / n,
        kappa=kappa,
        confusion=dict(
            sorted(Counter(f"{h}→{j}" for h, j in zip(human, judge, strict=True)).items())
        ),
    )
