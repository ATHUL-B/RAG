"""Parent-child ("small-to-big") retrieval (study guide Parts 4 and 12; HLD 023).

Small chunks are precise to *search*: one 200-token chunk is about one table row or one
paragraph, so its embedding isn't diluted by its neighbours. But small chunks are poor to
*read*: HLD 022 showed llama3.1:8b answering much worse from 200-token fragments (numeric
13/30 → 5/31 at the same prompt budget). Parent-child does both: search the small
**child** chunks, then return the larger **parent** chunk that contains the best child.

Here the parents are exactly the default 400-token chunks. Recall, the reranker's grade
cache and the prompt size are therefore unchanged and directly comparable; only *how
parents are found* changes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from rag.domain import Chunk, Query, ScoredChunk
from rag.retrieval.base import RetrievalResult, Retriever

STAGE = "parent"
CHILD_STAGE = "children"


def map_children_to_parents(
    children: Iterable[Chunk], parents: Iterable[Chunk]
) -> dict[str, Chunk]:
    """child chunk ID → the parent chunk (same document) with the largest character overlap.

    Both chunkers split the same document text, so offsets are comparable. A child that
    overlaps no parent (shouldn't happen with page-bounded chunkers on the same text) is
    left out, and the retriever skips it.
    """
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for parent in parents:
        by_doc[parent.doc_id].append(parent)
    for doc_parents in by_doc.values():
        doc_parents.sort(key=lambda c: c.char_start)
    mapping: dict[str, Chunk] = {}
    for child in children:
        best, best_overlap = None, 0
        for parent in by_doc.get(child.doc_id, ()):
            if parent.char_start >= child.char_end:
                break  # sorted: no later parent can overlap
            overlap = min(child.char_end, parent.char_end) - max(
                child.char_start, parent.char_start
            )
            if overlap > best_overlap:
                best, best_overlap = parent, overlap
        if best is not None:
            mapping[child.chunk_id] = best
    return mapping


class ParentChildRetriever(Retriever):
    """Search children, return their parents ranked by each parent's best child.

    Fetches ``k * fanout`` children, because several children often share one parent.
    The parent's score is its best child's score. The child list is kept in
    ``stage_candidates["children"]``, so a trace shows which small chunk found the parent.
    """

    def __init__(
        self, children: Retriever, parent_of: Mapping[str, Chunk], fanout: int = 4
    ) -> None:
        if fanout < 1:
            raise ValueError("fanout must be >= 1")
        self._children = children
        self._parent_of = parent_of
        self._fanout = fanout

    def retrieve(self, query: Query, k: int) -> RetrievalResult:
        result = self._children.retrieve(query, k * self._fanout)
        parents: list[ScoredChunk] = []
        seen: set[str] = set()
        for child in result.candidates:
            parent = self._parent_of.get(child.chunk.chunk_id)
            if parent is None or parent.chunk_id in seen:
                continue
            seen.add(parent.chunk_id)
            parents.append(
                ScoredChunk(chunk=parent, score=child.score, rank=len(parents) + 1, stage=STAGE)
            )
            if len(parents) == k:
                break
        return result.model_copy(
            update={
                "candidates": tuple(parents),
                "stage_candidates": {**result.stage_candidates, CHILD_STAGE: result.candidates},
            }
        )
