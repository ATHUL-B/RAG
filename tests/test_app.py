from pathlib import Path

from rag.app import build_llm_reranker, build_planner, check_local_models, local_models
from rag.config import (
    AppConfig,
    GenerationConfig,
    LLMPlannerConfig,
    LLMProvider,
    RerankConfig,
    RerankerKind,
    RetrievalConfig,
    RewriteConfig,
)
from rag.retrieval.llm_planner import LLMQueryPlanner
from rag.retrieval.llm_reranker import LLMPointwiseReranker


def _config(**planner: object) -> AppConfig:
    rewrite = RewriteConfig(llm_planner=LLMPlannerConfig.model_validate(planner))
    return AppConfig(experiment_name="x", retrieval=RetrievalConfig(rewrite=rewrite))


def test_no_planner_unless_enabled() -> None:
    assert build_planner(_config()) is None


def test_planner_is_built_lazily_without_contacting_ollama(tmp_path: Path) -> None:
    config = _config(enabled=True, model_id="llama3.1:8b")
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"data_dir": tmp_path})}
    )
    planner = build_planner(config)
    assert isinstance(planner, LLMQueryPlanner)
    assert planner.calls == 0  # nothing was sent anywhere
    assert not (tmp_path / "cache").exists()  # the cache file is written only on a new plan


def test_llm_reranker_only_for_enabled_llm_kind(tmp_path: Path) -> None:
    def config(enabled: bool, kind: RerankerKind) -> AppConfig:
        rerank = RerankConfig(enabled=enabled, kind=kind)
        base = AppConfig(experiment_name="x", retrieval=RetrievalConfig(rerank=rerank))
        return base.model_copy(
            update={"paths": base.paths.model_copy(update={"data_dir": tmp_path})}
        )

    assert build_llm_reranker(config(False, RerankerKind.LLM)) is None
    assert build_llm_reranker(config(True, RerankerKind.CROSS_ENCODER)) is None
    reranker = build_llm_reranker(config(True, RerankerKind.LLM))
    assert isinstance(reranker, LLMPointwiseReranker) and reranker.calls == 0


def test_local_models_lists_every_ollama_dependency_once() -> None:
    assert local_models(AppConfig(experiment_name="x")) == []
    rewrite = RewriteConfig(llm_planner=LLMPlannerConfig(enabled=True))
    rerank = RerankConfig(enabled=True, kind=RerankerKind.LLM)
    config = AppConfig(
        experiment_name="x",
        retrieval=RetrievalConfig(rewrite=rewrite, rerank=rerank),
        generation=GenerationConfig(provider=LLMProvider.OLLAMA, model_id="qwen2.5:7b"),
    )
    assert local_models(config) == [
        ("llama3.1:8b", "http://localhost:11434"),  # planner and reranker share it: once
        ("qwen2.5:7b", "http://localhost:11434"),
    ]
    checked: list[tuple[str, str]] = []
    check_local_models(config, check=lambda model, url: checked.append((model, url)))
    assert checked == local_models(config)
