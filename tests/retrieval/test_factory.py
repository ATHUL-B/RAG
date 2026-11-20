import pytest

from rag.config import (
    BM25Config,
    FilterConfig,
    FilterField,
    LLMPlannerConfig,
    RerankConfig,
    RetrievalConfig,
    RetrievalMode,
    RewriteConfig,
)
from rag.domain import Chunk
from rag.errors import ConfigError
from rag.indexing.vector_store import ExactVectorStore
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.factory import build_bm25_index, build_reranker, build_retriever
from rag.retrieval.filtered import FilteredRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.multi_query import MultiQueryRetriever
from rag.retrieval.rerankers import CrossEncoderReranker
from rag.retrieval.reranking import RerankingRetriever
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


def test_rerank_wraps_the_first_stage_and_filters_wrap_everything() -> None:
    config = RetrievalConfig(rerank=RerankConfig(enabled=True), filters=FilterConfig(enabled=True))
    reranker = build_reranker(RerankConfig())  # lazy: no model is loaded here
    retriever = build_retriever(config, EMBEDDER, STORE, {}, reranker=reranker)
    assert isinstance(retriever, FilteredRetriever)
    assert isinstance(
        build_retriever(
            config.model_copy(update={"filters": FilterConfig()}),
            EMBEDDER,
            STORE,
            {},
            reranker=reranker,
        ),
        RerankingRetriever,
    )
    assert isinstance(reranker, CrossEncoderReranker)


def test_rewrite_sits_inside_filters_so_filters_come_from_the_original_question() -> None:
    rewrite = RewriteConfig(strip_boilerplate=True)
    plain = build_retriever(RetrievalConfig(rewrite=rewrite), EMBEDDER, STORE, {})
    assert isinstance(plain, MultiQueryRetriever)
    config = RetrievalConfig(rewrite=rewrite, filters=FilterConfig(enabled=True))
    assert isinstance(build_retriever(config, EMBEDDER, STORE, {}), FilteredRetriever)


def test_rewrite_is_off_by_default() -> None:
    assert not RewriteConfig().enabled
    assert RewriteConfig(expand_abbreviations=True).enabled


def test_llm_planner_needs_an_injected_planner() -> None:
    rewrite = RewriteConfig(llm_planner=LLMPlannerConfig(enabled=True))
    assert rewrite.enabled and not rewrite.rules_enabled
    with pytest.raises(ConfigError, match="no planner"):
        build_retriever(RetrievalConfig(rewrite=rewrite), EMBEDDER, STORE, {})


def test_llm_planner_alone_or_with_rules_is_composed() -> None:
    from rag.retrieval.rewriting import RuleBasedRewriter

    planner = RuleBasedRewriter()  # any QueryRewriter stands in for the LLM planner
    for rules in (False, True):
        rewrite = RewriteConfig(strip_boilerplate=rules, llm_planner=LLMPlannerConfig(enabled=True))
        retriever = build_retriever(
            RetrievalConfig(rewrite=rewrite), EMBEDDER, STORE, {}, planner=planner
        )
        assert isinstance(retriever, MultiQueryRetriever)
