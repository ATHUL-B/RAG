"""HNSW approximate nearest-neighbour search, written out (guide Parts 5 and 14; HLD 028).

Exact search compares the query with every vector: O(n) per query. HNSW (Malkov &
Yashunin, 2016) instead walks a graph in which each vector is linked to a few close
neighbours, so a query only touches a small part of the corpus.

- **Layers.** Each vector gets a random level ``floor(-ln(U) * mL)``, ``mL = 1 / ln(M)``:
  every vector is on layer 0, about 1/M of them on layer 1, 1/M² on layer 2, and so on.
  The sparse upper layers act like an express road: a few long hops get near the query.
  The dense layer 0 then finds the actual neighbours.
- **Search.** Greedy descent on the upper layers (beam width 1), then a beam search of
  width ``ef`` on layer 0. A larger ``ef`` visits more nodes: higher recall, slower query.
- **Build.** Each new vector is searched for (beam ``ef_construction``) and linked to up
  to ``M`` neighbours (``2M`` on layer 0). Neighbours are picked with the paper's
  heuristic: skip a candidate that is closer to an already-picked neighbour than to the
  new vector. That keeps links pointing in different directions, so the graph doesn't
  get stuck inside one tight cluster.
- **Filters.** With an ``allowed`` mask the walk still goes through every node, but only
  allowed ones enter the results. A selective filter makes the walk wander until it
  finds ``ef`` allowed nodes, so callers should search a small allowed set exactly
  (:class:`HNSWVectorStore`).

Similarity is the dot product of L2-normalized vectors (cosine), as in the exact store.
This is a teaching implementation in Python: correct and inspectable, but its build is
~100x slower than faiss / hnswlib, which the index-recall script uses as a cross-check.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from rag.domain import Chunk, Filters, ScoredChunk
from rag.indexing.base import Matrix, Vector, VectorStore
from rag.indexing.vector_store import ExactVectorStore

Mask = npt.NDArray[np.bool_]
_STAGE = "dense"


class HNSWGraph:
    """The graph over a fixed matrix of L2-normalized vectors (row i = node i)."""

    def __init__(
        self, vectors: Matrix, m: int = 16, ef_construction: int = 100, seed: int = 0
    ) -> None:
        if m < 2:
            raise ValueError("m must be >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        self._v = np.ascontiguousarray(vectors, dtype=np.float32)
        self._m = m
        self._m0 = 2 * m  # layer 0 holds every node, so it gets more links (as in the paper)
        self._ml = 1.0 / math.log(m)
        self._ef_construction = ef_construction
        self._rng = np.random.default_rng(seed)
        self._links: list[dict[int, list[int]]] = []  # layer -> node -> neighbours
        self._entry = -1
        self.distance_computations = 0
        """Vectors compared with a query so far (the cost HNSW saves vs exact search)."""
        for node in range(len(self._v)):
            self._insert(node)

    def __len__(self) -> int:
        return len(self._v)

    @property
    def max_level(self) -> int:
        return len(self._links) - 1

    def layer_sizes(self) -> list[int]:
        return [len(layer) for layer in self._links]

    def neighbours(self, node: int, layer: int = 0) -> list[int]:
        return list(self._links[layer].get(node, []))

    # ------------------------------------------------------------------ search

    def search(
        self, query: Vector, k: int, ef: int, allowed: Mask | None = None
    ) -> list[tuple[float, int]]:
        """Up to ``k`` (similarity, node) pairs, best first. ``ef`` is raised to ``k``."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._entry < 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        entry = self._entry
        for layer in range(self.max_level, 0, -1):  # express layers: greedy, beam 1
            entry = self._search_layer(q, [entry], 1, layer)[0][1]
        return self._search_layer(q, [entry], max(ef, k), 0, allowed)[:k]

    def _search_layer(
        self, q: Vector, entries: list[int], ef: int, layer: int, allowed: Mask | None = None
    ) -> list[tuple[float, int]]:
        """Beam search on one layer (paper, Algorithm 2). Returns results best first.

        ``candidates`` is a max-heap of nodes still to expand (stored negated, since heapq
        is a min-heap). ``results`` is a min-heap of the ``ef`` best (allowed) nodes, so its
        worst one sits on top. The walk stops when the best unexpanded candidate is worse
        than the worst result: nothing left can improve the beam.
        """
        graph = self._links[layer]
        sims = self._v[entries] @ q
        self.distance_computations += len(entries)
        visited = set(entries)
        candidates = [(-float(s), e) for s, e in zip(sims, entries, strict=True)]
        heapq.heapify(candidates)
        results = [(float(s), e) for s, e in zip(sims, entries, strict=True) if _ok(allowed, e)]
        heapq.heapify(results)
        while candidates:
            negative, node = heapq.heappop(candidates)
            if len(results) >= ef and -negative < results[0][0]:
                break
            fresh = [n for n in graph.get(node, ()) if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            fresh_sims = self._v[fresh] @ q
            self.distance_computations += len(fresh)
            for sim, n in zip(fresh_sims.tolist(), fresh, strict=True):
                if len(results) < ef or sim > results[0][0]:
                    heapq.heappush(candidates, (-sim, n))
                    if _ok(allowed, n):
                        heapq.heappush(results, (sim, n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return sorted(results, reverse=True)

    # ------------------------------------------------------------------ build

    def _insert(self, node: int) -> None:
        """Paper, Algorithm 1."""
        level = int(-math.log(1.0 - self._rng.random()) * self._ml)
        q = self._v[node]
        if self._entry < 0:
            self._links = [{node: []} for _ in range(level + 1)]
            self._entry = node
            return
        entry = self._entry
        for layer in range(self.max_level, level, -1):
            entry = self._search_layer(q, [entry], 1, layer)[0][1]
        entries = [entry]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entries, self._ef_construction, layer)
            limit = self._m0 if layer == 0 else self._m
            picked = self._select(found, self._m)
            graph = self._links[layer]
            graph[node] = picked
            for n in picked:  # links are two-way; shrink a neighbour that is over its limit
                graph[n].append(node)
                if len(graph[n]) > limit:
                    sims = self._v[graph[n]] @ self._v[n]
                    ranked = sorted(zip(sims.tolist(), graph[n], strict=True), reverse=True)
                    graph[n] = self._select(ranked, limit)
            entries = [n for _, n in found]
        if level > self.max_level:
            self._links.extend({node: []} for _ in range(level - self.max_level))
            self._entry = node

    def _select(self, ranked: list[tuple[float, int]], m: int) -> list[int]:
        """Neighbour heuristic (paper, Algorithm 4): ``ranked`` is (similarity to the
        base, node), best first. Keep a candidate only if it is closer to the base than
        to every neighbour already kept."""
        kept: list[int] = []
        for sim, candidate in ranked:
            if len(kept) >= m:
                break
            if not kept or float(np.max(self._v[kept] @ self._v[candidate])) < sim:
                kept.append(candidate)
        return kept


def _ok(allowed: Mask | None, node: int) -> bool:
    return allowed is None or bool(allowed[node])


class HNSWVectorStore(VectorStore):
    """A :class:`VectorStore` that searches an :class:`HNSWGraph` over an exact store.

    Filters are pre-filters (Part 8), as in the exact store. When few chunks pass the
    filter (``exact_below``), it searches them exactly: comparing ~1k vectors costs less
    than walking a graph that is mostly filtered out, and it is always correct. This is
    the same fallback production vector databases use for selective filters.
    """

    def __init__(
        self,
        exact: ExactVectorStore,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        exact_below: int = 2000,
        seed: int = 0,
    ) -> None:
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")
        self._exact = exact
        self._graph = HNSWGraph(exact.vectors, m=m, ef_construction=ef_construction, seed=seed)
        self._ef_search = ef_search
        self._exact_below = exact_below
        self.exact_fallbacks = 0

    @property
    def graph(self) -> HNSWGraph:
        return self._graph

    @property
    def model_id(self) -> str:
        return self._exact.model_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._exact.chunks

    def __len__(self) -> int:
        return len(self._exact)

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        raise NotImplementedError("HNSWVectorStore is built from a loaded exact index")

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        allowed = self._exact.filter_mask(filters) if filters else None
        if allowed is not None and int(allowed.sum()) <= self._exact_below:
            self.exact_fallbacks += 1
            return self._exact.search_within(vector, k, allowed)
        hits = self._graph.search(vector, k, self._ef_search, allowed)
        chunks = self._exact.chunks
        return [
            ScoredChunk(chunk=chunks[node], score=sim, rank=rank, stage=_STAGE)
            for rank, (sim, node) in enumerate(hits, start=1)
        ]
