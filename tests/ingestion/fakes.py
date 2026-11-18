"""Deterministic token counters for chunker tests (no tokenizer download)."""

from rag.ingestion.tokenization import TokenCounter


class WordCounter(TokenCounter):
    """One token per whitespace-separated word."""

    @property
    def name(self) -> str:
        return "words"

    def count(self, text: str) -> int:
        return len(text.split())


class CharCounter(TokenCounter):
    """One token per non-whitespace character: forces hard splits of unbroken text."""

    @property
    def name(self) -> str:
        return "chars"

    def count(self, text: str) -> int:
        return sum(not ch.isspace() for ch in text)
