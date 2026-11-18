"""Token counting behind an interface (chunk sizes and model limits are in tokens)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import tiktoken


class TokenCounter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Identifies the tokenizer; part of the chunker fingerprint."""

    @abstractmethod
    def count(self, text: str) -> int: ...


class TiktokenCounter(TokenCounter):
    """OpenAI BPE encodings (e.g. ``cl100k_base``); a reasonable proxy for most embedders.

    The encoding file is downloaded on first use and cached by tiktoken.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name

    @property
    def name(self) -> str:
        return f"tiktoken-{self._encoding_name}"

    @cached_property
    def _encoding(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self._encoding_name)

    def count(self, text: str) -> int:
        # Treat special-token text like "<|endoftext|>" in documents as plain text.
        return len(self._encoding.encode(text, disallowed_special=()))
