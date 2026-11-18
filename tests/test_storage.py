from pathlib import Path

import pytest

from rag.domain import Document, Page
from rag.errors import StorageError
from rag.storage import JsonArtifactStore, atomic_write_bytes, validate_key


@pytest.fixture
def store(tmp_path: Path) -> JsonArtifactStore[Document]:
    return JsonArtifactStore(tmp_path / "docs", Document)


def _doc(doc_id: str = "A_2020") -> Document:
    return Document(
        doc_id=doc_id, source_uri="u", pages=(Page(page_index=0, text="hi"),), metadata={"y": 1}
    )


def test_round_trips_a_model(store: JsonArtifactStore[Document]) -> None:
    doc = _doc()
    _ = doc.text  # populate cached_property: must not leak into the artifact
    path = store.save("A_2020", doc)
    assert path == store.root / "A_2020.json"
    assert '"text": "hi"' in path.read_text()
    assert store.load("A_2020") == doc
    assert store.exists("A_2020")


def test_keys_lists_saved_items_sorted(store: JsonArtifactStore[Document]) -> None:
    assert store.keys() == []
    store.save("B", _doc("B"))
    store.save("A", _doc("A"))
    assert store.keys() == ["A", "B"]


def test_missing_and_corrupt_artifacts_raise_storage_error(
    store: JsonArtifactStore[Document],
) -> None:
    with pytest.raises(StorageError, match="no artifact"):
        store.load("missing")
    atomic_write_bytes(store.path("bad"), b'{"doc_id": ""}')
    with pytest.raises(StorageError, match="corrupt"):
        store.load("bad")


@pytest.mark.parametrize("key", ["../x", "a/b", "a.b", ""])
def test_unsafe_keys_are_rejected(key: str) -> None:
    with pytest.raises(StorageError, match="unsafe"):
        validate_key(key)


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "f.bin"
    atomic_write_bytes(target, b"data")
    assert target.read_bytes() == b"data"
    assert list(target.parent.iterdir()) == [target]
