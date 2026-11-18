from pathlib import Path

import pytest

from rag.domain import Chunk, ChunkSet
from rag.indexing.index_stage import IndexStage
from rag.storage import JsonArtifactStore
from support.embedders import HashingEmbedder


def _chunk(doc_id: str, i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{i}",
        doc_id=doc_id,
        text=text,
        char_start=i * 100,
        char_end=i * 100 + len(text),
        page_start=0,
        page_end=0,
    )


class TickClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 2.0
        return self.now


@pytest.fixture
def chunk_sets(tmp_path: Path) -> JsonArtifactStore[ChunkSet]:
    store = JsonArtifactStore(tmp_path, ChunkSet)
    texts = {"A": ["alpha beta", "gamma delta epsilon zeta"], "B": ["eta", "theta", "iota"]}
    for doc_id, chunk_texts in texts.items():
        chunks = tuple(_chunk(doc_id, i, t) for i, t in enumerate(chunk_texts))
        store.save(doc_id, ChunkSet(doc_id=doc_id, chunker_fingerprint="fp", chunks=chunks))
    return store


def test_embeds_all_chunks_in_batches(chunk_sets: JsonArtifactStore[ChunkSet]) -> None:
    embedder = HashingEmbedder(max_tokens=3)
    stage = IndexStage(embedder, chunk_sets, batch_size=2, clock=TickClock())
    store, report = stage.build(chunk_sets.ids())

    assert len(store) == 5
    assert store.model_id == embedder.model_id
    assert embedder.document_calls == 3  # batches of 2, 2, 1
    assert report.documents == 2
    assert report.chunks == 5
    assert report.truncated_chunk_ids == ["A-1"]  # 4 words > max_tokens=3
    assert report.seconds == 2.0
    assert report.chunks_per_second == 2.5


def test_rejects_invalid_batch_size(chunk_sets: JsonArtifactStore[ChunkSet]) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        IndexStage(HashingEmbedder(), chunk_sets, batch_size=0)
