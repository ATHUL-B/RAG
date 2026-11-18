import pytest

from rag.config import BM25Config, FilterConfig, FilterField, RetrievalConfig, RetrievalMode
from rag.domain import Chunk
from rag.errors import ConfigError
from rag.indexing.vector_store import ExactVectorStore
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.factory import build_bm25_index, build_retriever
from rag.retrieval.filtered import FilteredRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
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


CHUNK = Chunk(
    chunk_id="c", doc_id="D", text="net income", char_start=0, char_end=10, page_start=0, page_end=0
)


def test_build_bm25_index_uses_config_and_chunks() -> None:
    index = build_bm25_index(BM25Config(k1=1.5, b=0.5), [CHUNK])
    assert len(index) == 1
    assert index.search("income", 1)[0].chunk.chunk_id == "c"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(RetrievalMode.BM25, KeywordRetriever), (RetrievalMode.HYBRID, HybridRetriever)],
)
def test_keyword_modes(mode: RetrievalMode, expected: type) -> None:
    index = build_bm25_index(BM25Config(), [CHUNK])
    retriever = build_retriever(RetrievalConfig(mode=mode), EMBEDDER, STORE, {}, index)
    assert isinstance(retriever, expected)


def test_keyword_modes_require_a_keyword_index() -> None:
    with pytest.raises(ConfigError, match="keyword index"):
        build_retriever(RetrievalConfig(mode=RetrievalMode.HYBRID), EMBEDDER, STORE, {})
