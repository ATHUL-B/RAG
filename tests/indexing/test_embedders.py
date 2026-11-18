import numpy as np
import pytest

from rag.config import EmbeddingConfig
from rag.indexing.embedders import SentenceTransformerEmbedder, resolve_device


def test_model_id_includes_revision_without_loading_model() -> None:
    assert SentenceTransformerEmbedder("org/model", revision="abc").model_id == "org/model@abc"
    assert SentenceTransformerEmbedder("org/model").model_id == "org/model"


def test_explicit_device_is_kept() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_device_resolves_to_a_real_device() -> None:
    assert resolve_device("auto") in {"mps", "cuda", "cpu"}


def test_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder("org/model", batch_size=0)


@pytest.mark.slow
def test_real_model_ranks_matching_passage_first() -> None:
    config = EmbeddingConfig()
    embedder = SentenceTransformerEmbedder(
        config.model_id, config.revision, config.query_prefix, device="cpu"
    )
    passages = embedder.embed_documents(
        [
            "Purchases of property, plant and equipment were $1,577 million in 2018.",
            "The company declared a quarterly dividend to shareholders.",
        ]
    )
    query = embedder.embed_query("What was 3M's capital expenditure in FY2018?")

    assert passages.shape == (2, 384)
    assert embedder.max_tokens == 512
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-4)
    assert passages[0] @ query > passages[1] @ query
    assert embedder.embed_documents([]).shape == (0, 384)
