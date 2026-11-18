"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher 2009; study guide Part 9; HLD 007).

    RRF(d) = Σ_lists  w_list / (k + rank_list(d))

Only ranks are used, never raw scores: cosine similarities and BM25 scores live on
different scales and cannot be added directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from rag.domain import Chunk, ScoredChunk
from rag.retrieval.base import Fuser

_STAGE: Final = "rrf"


class ReciprocalRankFusion(Fuser):
    def __init__(self, k: int = 60, weights: Mapping[str, float] | None = None) -> None:
        if k < 0:
            raise ValueError("k must be >= 0")
        if weights and any(w < 0 for w in weights.values()):
            raise ValueError("weights must be >= 0")
        self._k = k
        self._weights = dict(weights or {})

    def fuse(self, rankings: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        scores: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        chunks: dict[str, Chunk] = {}  # insertion order = first appearance
        for ranking in rankings:
            for candidate in ranking:
                chunk_id = candidate.chunk.chunk_id
                weight = self._weights.get(candidate.stage, 1.0)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (self._k + candidate.rank)
                best_rank[chunk_id] = min(best_rank.get(chunk_id, candidate.rank), candidate.rank)
                chunks.setdefault(chunk_id, candidate.chunk)

        appearance = {chunk_id: i for i, chunk_id in enumerate(chunks)}
        order = sorted(chunks, key=lambda c: (-scores[c], best_rank[c], appearance[c]))[:k]
        return [
            ScoredChunk(chunk=chunks[c], score=scores[c], rank=rank, stage=_STAGE)
            for rank, c in enumerate(order, start=1)
        ]
