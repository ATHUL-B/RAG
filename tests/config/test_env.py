import os
from pathlib import Path

import pytest

from rag.config.env import load_env_file


def test_loads_unset_variables_and_never_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nRAG_TEST_A=one\nexport RAG_TEST_B='two'\nRAG_TEST_C=\"three\"\nnot a pair\n"
    )
    monkeypatch.setenv("RAG_TEST_C", "from-shell")
    for key in ("RAG_TEST_A", "RAG_TEST_B"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_env_file(env)

    assert loaded == ["RAG_TEST_A", "RAG_TEST_B"]
    assert os.environ["RAG_TEST_A"] == "one"
    assert os.environ["RAG_TEST_B"] == "two"
    assert os.environ["RAG_TEST_C"] == "from-shell"
    for key in loaded:
        monkeypatch.delenv(key)


def test_missing_file_loads_nothing(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") == []
