"""Retrieval metrics (study guide, Part 13; design in docs/hld/001-retrieval-metrics.md).

Inputs are deliberately generic:

- ``ranked``: IDs of retrieved items, best first. May contain repeats (two chunks
  that cover the same evidence page map to the same ID).
- ``qrels``: relevance grade per ID ("query relevance judgments"). A grade > 0 means
  relevant; binary labels use grade 1.

All functions are pure. Queries with no relevant items (unanswerable) have no
defined recall or nDCG, so those functions raise ``ValueError``; callers skip them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

Qrels = Mapping[str, int]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _relevant_ids(qrels: Qrels) -> set[str]:
    return {doc_id for doc_id, grade in qrels.items() if grade > 0}


def _require_relevant(qrels: Qrels) -> set[str]:
    relevant = _relevant_ids(qrels)
    if not relevant:
        raise ValueError("metric undefined: query has no relevant items (unanswerable)")
    return relevant


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """Share of the relevant items that appear in the top ``k`` (repeats count once).

    The question it answers: "did the evidence reach the candidate list at all?"
    """
    _check_k(k)
    relevant = _require_relevant(qrels)
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """Share of the top ``k`` positions that hold a relevant item.

    Divides by ``k`` even if fewer items were returned. "How much of what the
    model will read is useful?" Low precision = noise in the prompt.
    """
    _check_k(k)
    relevant = _relevant_ids(qrels)
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / k


def hit_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """1.0 if at least one relevant item is in the top ``k``, else 0.0 (success@k)."""
    _check_k(k)
    relevant = _relevant_ids(qrels)
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], qrels: Qrels) -> float:
    """1 / rank of the first relevant item (ranks start at 1); 0.0 if none.

    Averaged over queries this is MRR: "how far down does the user have to look?"
    """
    relevant = _relevant_ids(qrels)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain: grade / log2(rank + 1), ranks starting at 1."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    """Normalized DCG: rewards putting the *most* relevant items *first*.

    - Gain of a position = its grade, only on the first occurrence of an ID (so
      repeats cannot push the score above 1).
    - The ideal ranking uses the best ``k`` grades in ``qrels``, including relevant
      items that were never retrieved, so missing evidence lowers the score.
    """
    _check_k(k)
    _require_relevant(qrels)
    seen: set[str] = set()
    gains: list[float] = []
    for doc_id in ranked[:k]:
        gains.append(0.0 if doc_id in seen else float(max(qrels.get(doc_id, 0), 0)))
        seen.add(doc_id)
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    return _dcg(gains) / _dcg(ideal)


def recall_at_token_budget(
    ranked: Sequence[str], token_counts: Sequence[int], qrels: Qrels, budget: int
) -> float:
    """Recall over the longest prefix of the ranking whose chunks fit in ``budget`` tokens.

    recall@k compares rankings at the same *number* of chunks, which is unfair across chunk
    sizes: five 800-token chunks give the generator four times the text of five 200-token
    chunks. The prompt is limited by tokens, not by chunk count, so this counts what fits
    in the same token budget (HLD 022). A chunk that doesn't fit ends the prefix; it isn't
    skipped, because the generator sees sources in rank order.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if len(ranked) != len(token_counts):
        raise ValueError("ranked and token_counts must have the same length")
    relevant = _require_relevant(qrels)
    used, kept = 0, []
    for item, tokens in zip(ranked, token_counts, strict=True):
        if used + tokens > budget:
            break
        used += tokens
        kept.append(item)
    return len(relevant & set(kept)) / len(relevant)
