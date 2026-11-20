from pathlib import Path

from rag.app import build_planner
from rag.config import AppConfig, LLMPlannerConfig, RetrievalConfig, RewriteConfig
from rag.retrieval.llm_planner import LLMQueryPlanner


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
