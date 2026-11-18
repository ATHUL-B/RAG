from rag.config import FilterConfig, FilterField, RetrievalConfig
from rag.indexing.vector_store import ExactVectorStore
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.factory import build_retriever
from rag.retrieval.filtered import FilteredRetriever
from support.embedders import HashingEmbedder

EMBEDDER = HashingEmbedder()
STORE = ExactVectorStore(EMBEDDER.model_id, EMBEDDER.dimension)


def test_filters_disabled_builds_plain_dense_retriever() -> None:
    assert isinstance(build_retriever(RetrievalConfig(), EMBEDDER, STORE, {}), DenseRetriever)


def test_filters_enabled_wraps_dense_retriever() -> None:
    config = RetrievalConfig(
        filters=FilterConfig(enabled=True, fields=(FilterField.COMPANY,), year_window=1)
    )
    assert isinstance(build_retriever(config, EMBEDDER, STORE, {"3M": ()}), FilteredRetriever)
