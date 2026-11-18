from pathlib import Path

from rag.config import AppConfig, EmbeddingConfig
from rag.indexing.factory import build_embedder, dense_index_dir, model_slug


def test_model_slug_is_filesystem_safe() -> None:
    assert model_slug("BAAI/bge-small-en-v1.5") == "baai-bge-small-en-v1-5"


def test_index_dir_is_keyed_by_chunker_and_model() -> None:
    config = AppConfig(experiment_name="x")
    assert dense_index_dir(config, "fp123") == Path(
        "data/indexes/dense/fp123__baai-bge-small-en-v1-5"
    )


def test_build_embedder_uses_pinned_revision() -> None:
    embedder = build_embedder(EmbeddingConfig())
    assert embedder.model_id == f"BAAI/bge-small-en-v1.5@{EmbeddingConfig().revision}"
