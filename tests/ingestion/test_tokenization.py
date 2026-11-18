import pytest

from rag.ingestion.tokenization import TiktokenCounter


def test_name_includes_encoding_without_loading_it() -> None:
    assert TiktokenCounter("cl100k_base").name == "tiktoken-cl100k_base"


@pytest.mark.slow
def test_counts_real_tokens_and_tolerates_special_token_text() -> None:
    counter = TiktokenCounter("cl100k_base")
    assert counter.count("") == 0
    assert counter.count("hello world") == 2
    assert counter.count("text with <|endoftext|> inside") > 0
