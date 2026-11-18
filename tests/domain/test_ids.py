import pytest

from rag.domain import config_fingerprint, make_chunk_id


def test_config_fingerprint_ignores_key_order() -> None:
    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})


def test_config_fingerprint_changes_with_values() -> None:
    assert config_fingerprint({"size": 400}) != config_fingerprint({"size": 800})


def test_chunk_id_is_deterministic() -> None:
    assert make_chunk_id("doc", 0, 10, "fp") == make_chunk_id("doc", 0, 10, "fp")


@pytest.mark.parametrize(
    "other",
    [("doc2", 0, 10, "fp"), ("doc", 1, 10, "fp"), ("doc", 0, 11, "fp"), ("doc", 0, 10, "fp2")],
)
def test_chunk_id_changes_with_any_input(other: tuple[str, int, int, str]) -> None:
    assert make_chunk_id("doc", 0, 10, "fp") != make_chunk_id(*other)


def test_chunk_id_rejects_empty_span() -> None:
    with pytest.raises(ValueError, match="char_end"):
        make_chunk_id("doc", 5, 5, "fp")
