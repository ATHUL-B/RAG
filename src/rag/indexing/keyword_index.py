"""BM25 over an in-memory inverted index (study guide, Part 9; HLD 007).

Written out in full rather than imported, so every part of the formula is visible:

    score(D, Q) = Σ_t IDF(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
    IDF(t)      = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from rag.domain import Chunk, Filters, ScoredChunk, matches
from rag.errors import IndexingError
from rag.indexing.base import KeywordIndex
from rag.indexing.text_analysis import analyze

_STAGE: Final = "bm25"

Analyzer = Callable[[str], list[str]]


def bm25_idf(n_chunks: int, n_containing: int) -> float:
    """Rare terms get large weights; the +1 keeps IDF positive for very common terms."""
    return math.log((n_chunks - n_containing + 0.5) / (n_containing + 0.5) + 1.0)


class BM25Index(KeywordIndex):
    def __init__(self, k1: float = 1.2, b: float = 0.75, analyzer: Analyzer = analyze) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        self._k1 = k1
        self._b = b
        self._analyzer = analyzer
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._postings: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]] = {}
        self._lengths: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    # ------------------------------------------------------------------ build

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks. Rebuilds the posting arrays, so add in large batches."""
        new_postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        new_lengths: list[int] = []
        for chunk in chunks:
            if chunk.chunk_id in self._ids:
                raise IndexingError(f"chunk {chunk.chunk_id} is already indexed")
            self._ids.add(chunk.chunk_id)
            position = len(self._chunks)
            self._chunks.append(chunk)
            terms = Counter(self._analyzer(chunk.indexed_text))
            new_lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                positions, frequencies = new_postings[term]
                positions.append(position)
                frequencies.append(frequency)

        for term, (positions, frequencies) in new_postings.items():
            old_positions, old_frequencies = self._postings.get(
                term, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
            )
            self._postings[term] = (
                np.concatenate([old_positions, np.asarray(positions, dtype=np.int64)]),
                np.concatenate([old_frequencies, np.asarray(frequencies, dtype=np.float32)]),
            )
        self._lengths = np.concatenate([self._lengths, np.asarray(new_lengths, dtype=np.float32)])

    # ------------------------------------------------------------------ search

    def search(self, text: str, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._chunks:
            return []
        scores = self.scores(text)
        if filters:
            allowed = np.fromiter(
                (matches(c.metadata, filters) for c in self._chunks),
                dtype=np.bool_,
                count=len(self._chunks),
            )
            scores = np.where(allowed, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank, stage=_STAGE)
            for rank, i in enumerate(order, start=1)
        ]

    def scores(self, text: str) -> npt.NDArray[np.float64]:
        """BM25 score of every indexed chunk for ``text`` (0 for chunks sharing no term)."""
        n_chunks = len(self._chunks)
        scores = np.zeros(n_chunks, dtype=np.float64)
        avgdl = float(self._lengths.mean()) or 1.0
        norm = self._k1 * (1.0 - self._b + self._b * self._lengths / avgdl)
        for term in dict.fromkeys(self._analyzer(text)):  # unique, order kept
            if term not in self._postings:
                continue
            positions, frequencies = self._postings[term]
            idf = bm25_idf(n_chunks, len(positions))
            scores[positions] += (
                idf * frequencies * (self._k1 + 1.0) / (frequencies + norm[positions])
            )
        return scores
