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


@pytest.mark.parametrize("name", ["filters.yaml", "filters-window1.yaml"])
def test_committed_ablation_configs_are_valid_and_differ_only_in_filters(name: str) -> None:
    baseline = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    ablation = load_config(REPO_ROOT / "configs" / name)
    assert ablation.retrieval.filters.enabled
    same = {"experiment_name": "x", "retrieval": None}
    assert baseline.model_copy(update=same) == ablation.model_copy(update=same)


def test_unknown_filter_field_is_rejected(tmp_path: Path) -> None:
    content = "experiment_name: x\nretrieval:\n  filters:\n    fields: [sector]\n"
    with pytest.raises(ConfigError, match="fields"):
        load_config(_write(tmp_path, content))
