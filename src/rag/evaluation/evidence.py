"""Evidence → chunk relevance (HLD 003).

Golden evidence is labelled at page level (doc + zero-based page). The *relevance unit*
is therefore an evidence page, and a retrieved chunk is relevant when it covers one.
This is the bridge between chunk rankings and the metrics in ``retrieval_metrics``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain import Chunk, GoldenItem, SourceSpan
from rag.evaluation.retrieval_metrics import Qrels


def evidence_unit(doc_id: str, page_index: int) -> str:
    return f"{doc_id}#p{page_index}"


def qrels_for(item: GoldenItem) -> Qrels:
    """Binary relevance: every distinct evidence page of the item is one relevant unit."""
    return {evidence_unit(span.doc_id, span.page_index): 1 for span in item.evidence}


def covers(chunk: Chunk, span: SourceSpan) -> bool:
    return chunk.doc_id == span.doc_id and chunk.page_start <= span.page_index <= chunk.page_end


def relevance_ids(ranked: Sequence[Chunk], item: GoldenItem) -> list[str]:
    """Translate a chunk ranking into relevance-unit IDs for this item.

    A chunk becomes the unit of the first evidence page it covers; otherwise it keeps its
    own ``chunk_id`` (never in the qrels, so non-relevant). Two chunks from the same
    evidence page yield the same unit, which the metrics handle (HLD 001).
    """
    ids: list[str] = []
    for chunk in ranked:
        span = next((s for s in item.evidence if covers(chunk, s)), None)
        ids.append(evidence_unit(span.doc_id, span.page_index) if span else chunk.chunk_id)
    return ids
