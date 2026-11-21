from pathlib import Path

import pytest

from rag.config import AppConfig, ChunkingStrategy, load_config
from rag.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_committed_baseline_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    assert config.experiment_name == "baseline"
    assert config.chunking.strategy is ChunkingStrategy.RECURSIVE


def test_defaults_fill_missing_sections(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, "experiment_name: x\n"))
    assert config == AppConfig(experiment_name="x")
    assert config.paths.raw_dir == Path("data/raw")


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(_write(tmp_path, "experiment_name: [unclosed\n"))


def test_non_mapping_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_typo_in_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="chunk_sise"):
        load_config(_write(tmp_path, "experiment_name: x\nchunking:\n  chunk_sise: 10\n"))


def test_overlap_must_be_smaller_than_chunk(tmp_path: Path) -> None:
    content = "experiment_name: x\nchunking:\n  chunk_size: 100\n  chunk_overlap: 100\n"
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_config(_write(tmp_path, content))


def test_evaluation_ks_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ks"):
        load_config(_write(tmp_path, "experiment_name: x\nevaluation:\n  ks: [0, 5]\n"))


def test_chunks_dir_is_namespaced_by_chunker_fingerprint() -> None:
    config = AppConfig(experiment_name="x")
    assert config.paths.chunks_dir("abc") == Path("data/processed/chunks/abc")


@pytest.mark.parametrize(
    "name",
    [
        "filters.yaml",
        "filters-window1.yaml",
        "bm25.yaml",
        "hybrid.yaml",
        "rerank-minilm.yaml",
        "rerank-bge.yaml",
        "dense-rerank-bge.yaml",
    ],
)
def test_committed_ablation_configs_differ_from_baseline_only_in_retrieval(name: str) -> None:
    baseline = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    ablation = load_config(REPO_ROOT / "configs" / name)
    assert ablation.retrieval.filters.enabled
    same = {"experiment_name": "x", "retrieval": None}
    assert baseline.model_copy(update=same) == ablation.model_copy(update=same)


@pytest.mark.parametrize(("name", "mode"), [("bm25.yaml", "bm25"), ("hybrid.yaml", "hybrid")])
def test_keyword_ablations_change_only_the_mode_vs_filters(name: str, mode: str) -> None:
    filters = load_config(REPO_ROOT / "configs" / "filters.yaml")
    ablation = load_config(REPO_ROOT / "configs" / name)
    assert ablation.retrieval.mode == mode
    assert ablation.retrieval.model_copy(update={"mode": filters.retrieval.mode}) == (
        filters.retrieval
    )


def test_unknown_filter_field_is_rejected(tmp_path: Path) -> None:
    content = "experiment_name: x\nretrieval:\n  filters:\n    fields: [sector]\n"
    with pytest.raises(ConfigError, match="fields"):
        load_config(_write(tmp_path, content))


@pytest.mark.parametrize(
    ("name", "base"),
    [
        ("rerank-minilm.yaml", "hybrid.yaml"),
        ("rerank-bge.yaml", "hybrid.yaml"),
        ("dense-rerank-bge.yaml", "filters.yaml"),
    ],
)
def test_rerank_ablations_change_only_rerank(name: str, base: str) -> None:
    before = load_config(REPO_ROOT / "configs" / base).retrieval
    after = load_config(REPO_ROOT / "configs" / name).retrieval
    assert after.rerank.enabled
    assert after.model_copy(update={"rerank": before.rerank}) == before


@pytest.mark.parametrize(
    ("name", "base", "section"),
    [
        ("pages.yaml", "filters.yaml", "chunking"),
        ("pages-context.yaml", "pages.yaml", "chunking"),
        ("pages-context-hybrid.yaml", "pages-context.yaml", "retrieval"),
        ("pages-context-rerank.yaml", "pages-context.yaml", "retrieval"),
    ],
)
def test_structural_ablations_change_one_section(name: str, base: str, section: str) -> None:
    before = load_config(REPO_ROOT / "configs" / base)
    after = load_config(REPO_ROOT / "configs" / name)
    same = {"experiment_name": "x", section: None}
    assert before.model_copy(update=same) == after.model_copy(update=same)
    assert getattr(before, section) != getattr(after, section)


@pytest.mark.parametrize(
    ("name", "base", "changed"),
    [
        ("planner-fy.yaml", "pages-context.yaml", {"filters"}),
        ("planner-forward.yaml", "pages-context.yaml", {"filters"}),
        ("planner-clean.yaml", "pages-context.yaml", {"rewrite"}),
        ("planner-abbrev.yaml", "pages-context.yaml", {"rewrite"}),
        ("planner-rules.yaml", "planner-clean.yaml", {"filters", "rewrite"}),
        ("planner-llm.yaml", "planner-rules.yaml", {"rewrite"}),
        ("planner-llm-w1-5.yaml", "planner-llm.yaml", {"rewrite"}),
        ("planner-llm-w2.yaml", "planner-llm.yaml", {"rewrite"}),
        ("planner-llm-w3.yaml", "planner-llm.yaml", {"rewrite"}),
        ("planner-llm-rerank-llm.yaml", "planner-llm.yaml", {"rerank"}),
        ("planner-llm-rerank-bge.yaml", "planner-llm.yaml", {"rerank"}),
    ],
)
def test_planner_ablations_change_only_filters_or_rewrite(
    name: str, base: str, changed: set[str]
) -> None:
    before = load_config(REPO_ROOT / "configs" / base)
    after = load_config(REPO_ROOT / "configs" / name)
    same = {"experiment_name": "x", "retrieval": None}
    assert before.model_copy(update=same) == after.model_copy(update=same)
    keep = {field: getattr(before.retrieval, field) for field in changed}
    assert after.retrieval.model_copy(update=keep) == before.retrieval
    assert all(getattr(after.retrieval, f) != getattr(before.retrieval, f) for f in changed)


@pytest.mark.parametrize("name", ["generation-claude.yaml", "generation-ollama.yaml"])
def test_generation_configs_reuse_the_default_retrieval(name: str) -> None:
    default = load_config(REPO_ROOT / "configs" / "planner-rules.yaml")
    generation = load_config(REPO_ROOT / "configs" / name)
    same = {"experiment_name": "x", "generation": None}
    assert default.model_copy(update=same) == generation.model_copy(update=same)


@pytest.mark.parametrize(("name", "weight"), [("w1-5", 1.5), ("w2", 2.0), ("w3", 3.0)])
def test_weight_ablations_change_only_the_original_weight(name: str, weight: float) -> None:
    before = load_config(REPO_ROOT / "configs" / "planner-llm.yaml").retrieval.rewrite
    after = load_config(REPO_ROOT / "configs" / f"planner-llm-{name}.yaml").retrieval.rewrite
    assert after.original_weight == weight
    assert after.model_copy(update={"original_weight": 1.0}) == before


def test_best_generation_config_only_swaps_in_the_best_retrieval() -> None:
    plain = load_config(REPO_ROOT / "configs" / "generation-ollama.yaml")
    best = load_config(REPO_ROOT / "configs" / "generation-ollama-best.yaml")
    retrieval = load_config(REPO_ROOT / "configs" / "planner-llm-rerank-llm.yaml").retrieval
    assert best.retrieval == retrieval
    same = {"experiment_name": "x", "retrieval": None}
    assert plain.model_copy(update=same) == best.model_copy(update=same)


def test_gateway_config_only_swaps_the_generator() -> None:
    ollama = load_config(REPO_ROOT / "configs" / "generation-ollama.yaml")
    gateway = load_config(REPO_ROOT / "configs" / "generation-gateway.yaml")
    same = {"experiment_name": "x", "generation": None}
    assert ollama.model_copy(update=same) == gateway.model_copy(update=same)
    assert gateway.generation.model_id.startswith("<")  # a placeholder the runner must set


@pytest.mark.parametrize(
    "base", ["planner-rules", "planner-llm-rerank-llm", "generation-ollama-best"]
)
def test_layout_configs_change_only_the_parser(base: str) -> None:
    before = load_config(REPO_ROOT / "configs" / f"{base}.yaml")
    after = load_config(REPO_ROOT / "configs" / f"{base}-layout.yaml")
    same = {"experiment_name": "x", "parsing": None}
    assert before.model_copy(update=same) == after.model_copy(update=same)
    assert (before.parsing.parser, after.parsing.parser) == ("pypdf", "pypdf-layout")
