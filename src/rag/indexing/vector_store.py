"""Exact (brute-force) vector search in NumPy (ADR 0003, HLD 004).

Every query is compared with every allowed chunk: slow at millions of vectors, but
always correct, which makes it the reference for measuring approximate indexes later.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, ValidationError

from rag.domain import Chunk, Filters, ScoredChunk, matches
from rag.errors import IndexingError, StorageError
from rag.indexing.base import Matrix, Vector, VectorStore
from rag.storage import atomic_write_bytes

VECTORS_FILE: Final = "vectors.npy"
CHUNKS_FILE: Final = "chunks.jsonl"
MANIFEST_FILE: Final = "manifest.json"
_NORM_TOLERANCE: Final = 1e-3
_STAGE: Final = "dense"


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    dimension: int
    count: int


class ExactVectorStore(VectorStore):
    """Cosine similarity over L2-normalized vectors (cosine == dot product)."""

    def __init__(self, model_id: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._model_id = model_id
        self._dimension = dimension
        self._chunks: list[Chunk] = []
        self._position: dict[str, int] = {}
        self._matrix: Matrix = np.zeros((0, dimension), dtype=np.float32)

    @property
    def model_id(self) -> str:
        return self._model_id

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Indexed chunks in insertion order (e.g. to know which evidence is indexed)."""
        return tuple(self._chunks)

    @property
    def vectors(self) -> Matrix:
        """Row i is the vector of ``chunks[i]`` (read-only view, e.g. to build an ANN index)."""
        view = self._matrix.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------ write

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        """Add chunks; a chunk whose ID already exists is replaced in place."""
        vectors = np.asarray(vectors, dtype=np.float32)
        self._check_vectors(vectors, expected_rows=len(chunks))
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise IndexingError("duplicate chunk_id within one upsert batch")
        new_rows: list[int] = []
        for row, chunk in enumerate(chunks):
            existing = self._position.get(chunk.chunk_id)
            if existing is None:
                self._position[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                new_rows.append(row)
            else:
                self._chunks[existing] = chunk
                self._matrix[existing] = vectors[row]
        if new_rows:
            self._matrix = np.vstack([self._matrix, vectors[new_rows]])

    def _check_vectors(self, vectors: Matrix, expected_rows: int) -> None:
        if vectors.ndim != 2 or vectors.shape != (expected_rows, self._dimension):
            raise IndexingError(
                f"expected vectors of shape ({expected_rows}, {self._dimension}), "
                f"got {vectors.shape}"
            )
        norms = np.linalg.norm(vectors, axis=1)
        if norms.size and not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise IndexingError("vectors must be L2-normalized (cosine similarity == dot)")

    # ------------------------------------------------------------------ read

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        return self.search_within(vector, k, self.filter_mask(filters) if filters else None)

    def search_within(
        self, vector: Vector, k: int, allowed: npt.NDArray[np.bool_] | None
    ) -> list[ScoredChunk]:
        """Exact top-k among the chunks ``allowed`` marks (all if None), for callers that
        already built the filter mask (the HNSW fallback): building it costs as much as
        the search itself (HLD 028)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self._dimension,):
            raise IndexingError(f"query has shape {query.shape}, expected ({self._dimension},)")
        candidates = np.flatnonzero(allowed) if allowed is not None else np.arange(len(self))
        if candidates.size == 0:
            return []
        scores = self._matrix[candidates] @ query
        top = _top_k_stable(scores, k)
        return [
            ScoredChunk(
                chunk=self._chunks[candidates[i]],
                score=float(scores[i]),
                rank=rank,
                stage=_STAGE,
            )
            for rank, i in enumerate(top, start=1)
        ]

    def filter_mask(self, filters: Filters) -> npt.NDArray[np.bool_]:
        """Pre-filter (Part 8): a chunk is allowed only if every filter matches."""
        return np.fromiter(
            (matches(c.metadata, filters) for c in self._chunks),
            dtype=np.bool_,
            count=len(self._chunks),
        )

    # ------------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self._matrix, allow_pickle=False)
        lines = "".join(chunk.model_dump_json() + "\n" for chunk in self._chunks)
        atomic_write_bytes(directory / CHUNKS_FILE, lines.encode("utf-8"))
        manifest = IndexManifest(
            model_id=self._model_id, dimension=self._dimension, count=len(self)
        )
        atomic_write_bytes(directory / MANIFEST_FILE, manifest.model_dump_json(indent=2).encode())

    @classmethod
    def load(cls, directory: Path, expected_model_id: str) -> ExactVectorStore:
        """Load an index; refuses one built with a different model (never mix models)."""
        try:
            manifest = IndexManifest.model_validate_json((directory / MANIFEST_FILE).read_bytes())
            matrix = np.load(directory / VECTORS_FILE, allow_pickle=False)
            with (directory / CHUNKS_FILE).open(encoding="utf-8") as handle:
                chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        except FileNotFoundError as exc:
            raise StorageError(f"no vector index at {directory}: {exc}") from exc
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise StorageError(f"corrupt vector index at {directory}: {exc}") from exc

        if manifest.model_id != expected_model_id:
            raise IndexingError(
                f"index at {directory} was built with {manifest.model_id}, "
                f"not {expected_model_id}: re-index instead of mixing models"
            )
        if len(chunks) != manifest.count or matrix.shape != (manifest.count, manifest.dimension):
            raise StorageError(f"vector index at {directory} is inconsistent with its manifest")
        store = cls(manifest.model_id, manifest.dimension)
        store.upsert(chunks, matrix)
        return store


def _top_k_stable(scores: npt.NDArray[np.float32], k: int) -> list[int]:
    """Indices of the k best scores, best first; equal scores keep insertion order.

    A full stable sort (O(n log n)) rather than ``argpartition``: exact and deterministic,
    and ~1 ms at our scale. An ANN index replaces this when scale demands it.
    """
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order]
