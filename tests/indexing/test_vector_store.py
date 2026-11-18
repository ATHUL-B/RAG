from pathlib import Path

import numpy as np
import pytest

from rag.domain import Chunk, Metadata
from rag.errors import IndexingError, StorageError
from rag.indexing.vector_store import CHUNKS_FILE, ExactVectorStore
from support.embedders import HashingEmbedder

EMBEDDER = HashingEmbedder(dimension=64)


def _chunk(chunk_id: str, text: str, **metadata: str | int) -> Chunk:
    meta: Metadata = dict(metadata)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"DOC_{metadata.get('year', 0)}",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=0,
        page_end=0,
        metadata=meta,
    )


CHUNKS = [
    _chunk("c1", "capital expenditure was 1577 million", company="3M", year=2018),
    _chunk("c2", "revenue grew in the industrial segment", company="3M", year=2018),
    _chunk("c3", "capital expenditure was 1373 million", company="3M", year=2017),
    _chunk("c4", "dividends paid to shareholders", company="AMD", year=2018),
]


@pytest.fixture
def store() -> ExactVectorStore:
    store = ExactVectorStore(EMBEDDER.model_id, EMBEDDER.dimension)
    store.upsert(CHUNKS, EMBEDDER.embed_documents([c.text for c in CHUNKS]))
    return store


def _search(store: ExactVectorStore, text: str, k: int, **filters: str | int) -> list[str]:
    hits = store.search(EMBEDDER.embed_query(text), k, dict(filters) or None)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.stage == "dense" for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    return [h.chunk.chunk_id for h in hits]


class TestSearch:
    def test_returns_most_similar_first(self, store: ExactVectorStore) -> None:
        assert _search(store, "capital expenditure", k=2) == ["c1", "c3"]

    def test_k_larger_than_index_returns_everything(self, store: ExactVectorStore) -> None:
        assert len(_search(store, "capital", k=50)) == 4

    def test_filters_apply_before_ranking(self, store: ExactVectorStore) -> None:
        """Part 8: the wrong year cannot take a top-k slot, even if it scores higher."""
        assert _search(store, "capital expenditure", k=1, year=2017) == ["c3"]
        assert _search(store, "capital expenditure", k=5, company="3M", year=2018) == ["c1", "c2"]

    def test_filter_matching_nothing_returns_empty(self, store: ExactVectorStore) -> None:
        assert _search(store, "capital", k=3, company="NOPE") == []

    def test_ties_keep_insertion_order(self) -> None:
        store = ExactVectorStore(EMBEDDER.model_id, EMBEDDER.dimension)
        same = [_chunk(f"t{i}", "identical text") for i in range(5)]
        store.upsert(same, EMBEDDER.embed_documents([c.text for c in same]))
        assert _search(store, "identical text", k=3) == ["t0", "t1", "t2"]

    def test_empty_store_returns_empty(self) -> None:
        assert ExactVectorStore("m", 64).search(EMBEDDER.embed_query("x"), 3) == []

    def test_rejects_bad_k_and_query_shape(self, store: ExactVectorStore) -> None:
        with pytest.raises(ValueError, match="k must be"):
            store.search(EMBEDDER.embed_query("x"), 0)
        with pytest.raises(IndexingError, match="shape"):
            store.search(np.ones(3, dtype=np.float32), 1)


class TestUpsert:
    def test_same_id_replaces_in_place(self, store: ExactVectorStore) -> None:
        replacement = _chunk("c4", "capital expenditure capital expenditure", company="AMD")
        store.upsert([replacement], EMBEDDER.embed_documents([replacement.text]))
        assert len(store) == 4
        assert [c.chunk_id for c in store.chunks] == ["c1", "c2", "c3", "c4"]
        assert store.chunks[3].text == replacement.text
        assert _search(store, "capital expenditure", k=1) == ["c4"]

    def test_rejects_wrong_shape_unnormalized_and_duplicate_ids(
        self, store: ExactVectorStore
    ) -> None:
        chunk = _chunk("x", "text")
        with pytest.raises(IndexingError, match="shape"):
            store.upsert([chunk], np.ones((1, 3), dtype=np.float32))
        with pytest.raises(IndexingError, match="normalized"):
            store.upsert([chunk], np.ones((1, 64), dtype=np.float32))
        with pytest.raises(IndexingError, match="duplicate"):
            store.upsert([chunk, chunk], EMBEDDER.embed_documents(["text", "text"]))

    def test_rejects_invalid_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            ExactVectorStore("m", 0)


class TestPersistence:
    def test_save_load_round_trip(self, store: ExactVectorStore, tmp_path: Path) -> None:
        store.save(tmp_path)
        loaded = ExactVectorStore.load(tmp_path, expected_model_id=EMBEDDER.model_id)
        assert len(loaded) == 4
        assert _search(loaded, "capital expenditure", k=2) == ["c1", "c3"]

    def test_refuses_index_built_with_another_model(
        self, store: ExactVectorStore, tmp_path: Path
    ) -> None:
        store.save(tmp_path)
        with pytest.raises(IndexingError, match="re-index"):
            ExactVectorStore.load(tmp_path, expected_model_id="other-model")

    def test_missing_or_inconsistent_index_raises_storage_error(
        self, store: ExactVectorStore, tmp_path: Path
    ) -> None:
        with pytest.raises(StorageError, match="no vector index"):
            ExactVectorStore.load(tmp_path / "missing", expected_model_id="m")
        store.save(tmp_path)
        lines = (tmp_path / CHUNKS_FILE).read_text().splitlines()
        (tmp_path / CHUNKS_FILE).write_text("\n".join(lines[:2]) + "\n")
        with pytest.raises(StorageError, match="inconsistent"):
            ExactVectorStore.load(tmp_path, expected_model_id=EMBEDDER.model_id)
