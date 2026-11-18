from pathlib import Path

import pytest

from rag.domain import ChunkSet, Document, Page
from rag.ingestion.chunk_stage import ChunkStage
from rag.ingestion.chunkers import RecursiveChunker
from rag.storage import JsonArtifactStore, atomic_write_bytes
from support.tokenizers import WordCounter


@pytest.fixture
def stores(tmp_path: Path) -> tuple[JsonArtifactStore[Document], JsonArtifactStore[ChunkSet]]:
    documents = JsonArtifactStore(tmp_path / "parsed", Document)
    for doc_id, text in {"A_1": "a b c\n\nd e", "B_1": "f g"}.items():
        pages = (Page(page_index=0, text=text),)
        documents.save(doc_id, Document(doc_id=doc_id, source_uri="u", pages=pages))
    return documents, JsonArtifactStore(tmp_path / "chunks", ChunkSet)


def test_chunks_and_saves_each_document(
    stores: tuple[JsonArtifactStore[Document], JsonArtifactStore[ChunkSet]],
) -> None:
    documents, chunk_sets = stores
    chunker = RecursiveChunker(WordCounter(), chunk_size=3)
    report = ChunkStage(chunker, documents, chunk_sets).run(documents.ids())

    assert report.chunked == ["A_1", "B_1"]
    assert report.chunks == 3
    saved = chunk_sets.load("A_1")
    assert saved.chunker_fingerprint == chunker.fingerprint
    assert [c.text for c in saved.chunks] == ["a b c", "d e"]


def test_skips_existing_and_records_failures(
    stores: tuple[JsonArtifactStore[Document], JsonArtifactStore[ChunkSet]],
) -> None:
    documents, chunk_sets = stores
    stage = ChunkStage(RecursiveChunker(WordCounter(), chunk_size=3), documents, chunk_sets)
    stage.run(["A_1"])
    atomic_write_bytes(documents.path("BAD_1"), b"{}")

    report = stage.run(["A_1", "BAD_1", "MISSING_1", "B_1"])

    assert report.skipped == ["A_1"]
    assert report.chunked == ["B_1"]
    assert set(report.failed) == {"BAD_1", "MISSING_1"}
