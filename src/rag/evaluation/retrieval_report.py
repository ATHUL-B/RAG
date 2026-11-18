"""Per-query retrieval scoring and aggregation, overall and per segment tag.

An overall average hides a collapse on one segment (study guide, Part 13), so
every report carries per-tag means with their counts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from rag.evaluation.retrieval_metrics import (
    Qrels,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

OVERALL = "overall"


class QueryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    tags: frozenset[str] = frozenset()
    scores: dict[str, float]


class SegmentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=1)
    means: dict[str, float]


class MetricsReport(BaseModel):
    """``segments[OVERALL]`` is the all-query mean; other keys are tags."""

    model_config = ConfigDict(frozen=True)

    segments: dict[str, SegmentSummary]

    @property
    def overall(self) -> SegmentSummary:
        return self.segments[OVERALL]


class RetrievalEvaluator:
    """Scores one query's ranking at a fixed set of cut-offs ``ks``."""

    def __init__(self, ks: Collection[int]) -> None:
        if not ks or min(ks) < 1:
            raise ValueError(f"ks must be non-empty and >= 1, got {ks}")
        self._ks = tuple(sorted(set(ks)))

    @property
    def ks(self) -> tuple[int, ...]:
        return self._ks

    def score(
        self, item_id: str, ranked: Sequence[str], qrels: Qrels, tags: Iterable[str] = ()
    ) -> QueryMetrics:
        """Raises ``ValueError`` for unanswerable queries (no relevant items)."""
        scores: dict[str, float] = {"mrr": reciprocal_rank(ranked, qrels)}
        for k in self._ks:
            scores[f"recall@{k}"] = recall_at_k(ranked, qrels, k)
            scores[f"precision@{k}"] = precision_at_k(ranked, qrels, k)
            scores[f"hit@{k}"] = hit_at_k(ranked, qrels, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels, k)
        return QueryMetrics(item_id=item_id, tags=frozenset(tags), scores=scores)


def aggregate(results: Sequence[QueryMetrics]) -> MetricsReport:
    """Mean of every metric overall and per tag."""
    if not results:
        raise ValueError("cannot aggregate zero results")
    groups: dict[str, list[QueryMetrics]] = defaultdict(list)
    for result in results:
        groups[OVERALL].append(result)
        for tag in result.tags:
            groups[tag].append(result)
    return MetricsReport(
        segments={name: _summarize(members) for name, members in sorted(groups.items())}
    )


def _summarize(results: Sequence[QueryMetrics]) -> SegmentSummary:
    names: Mapping[str, None] = dict.fromkeys(n for r in results for n in r.scores)
    missing = [r.item_id for r in results if r.scores.keys() != names.keys()]
    if missing:
        raise ValueError(f"results computed with different metrics: {missing}")
    means = {name: fmean(r.scores[name] for r in results) for name in names}
    return SegmentSummary(count=len(results), means=means)
