"""Recursive chunking with exact source offsets (study guide, Part 4; HLD 003)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from rag.domain import Chunk, Document, config_fingerprint, make_chunk_id
from rag.ingestion.base import Chunker
from rag.ingestion.tokenization import TokenCounter

DEFAULT_SEPARATORS: Final = ("\n\n", "\n", ". ", " ")
"""Coarsest first: paragraphs (and page breaks), lines, sentences, words."""


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int
    end: int
    tokens: int


class RecursiveChunker(Chunker):
    """Split on the coarsest separator that works, then pack pieces up to ``chunk_size``.

    Offsets always point into ``Document.text``; chunk text is never edited, only
    trimmed by moving offsets, so every chunk can be traced back to its source.
    """

    STRATEGY: Final = "recursive"

    def __init__(
        self,
        token_counter: TokenCounter,
        chunk_size: int,
        chunk_overlap: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        respect_pages: bool = False,
        source: str = "pypdf",
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if any(not sep for sep in separators):
            raise ValueError("separators must be non-empty strings")
        self._counter = token_counter
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = tuple(separators)
        self._respect_pages = respect_pages
        self._source = source

    @cached_property
    def fingerprint(self) -> str:
        settings: dict[str, object] = {
            "strategy": self.STRATEGY,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "tokenizer": self._counter.name,
            "separators": list(self._separators),
        }
        if self._respect_pages:  # only when on: existing fingerprints (and indexes) stay valid
            settings["respect_pages"] = True
        if self._source != "pypdf":  # the parser that produced the text (HLD 021)
            settings["source"] = self._source
        return config_fingerprint(settings)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []
        if self._respect_pages:  # chunks never cross a page boundary (HLD 009)
            regions = [(start, end) for _, start, end in document.page_spans() if start < end]
        else:
            regions = [(0, len(text))]
        chunks: list[Chunk] = []
        for region_start, region_end in regions:
            pieces = self._split(text, region_start, region_end, self._separators)
            for start, end in self._merge(pieces):
                start, end = _trim(text, start, end)
                if start < end:
                    chunks.append(self._make_chunk(document, start, end))
        return chunks

    # ------------------------------------------------------------------ split

    def _split(self, text: str, start: int, end: int, separators: tuple[str, ...]) -> list[_Piece]:
        tokens = self._counter.count(text[start:end])
        if tokens <= self._chunk_size:
            return [_Piece(start, end, tokens)]
        if not separators:
            return self._hard_split(text, start, end, tokens)
        separator, finer = separators[0], separators[1:]
        pieces: list[_Piece] = []
        for piece_start, piece_end in _split_keep_separator(text, start, end, separator):
            pieces.extend(self._split(text, piece_start, piece_end, finer))
        return pieces

    def _hard_split(self, text: str, start: int, end: int, tokens: int) -> list[_Piece]:
        """Last resort for unbroken text: fixed character windows sized from the token ratio."""
        chars_per_token = (end - start) / tokens
        window = max(1, int(self._chunk_size * chars_per_token * 0.9))
        return [
            _Piece(s, min(s + window, end), self._counter.count(text[s : min(s + window, end)]))
            for s in range(start, end, window)
        ]

    # ------------------------------------------------------------------ merge

    def _merge(self, pieces: list[_Piece]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        current: list[_Piece] = []
        current_tokens = 0
        for piece in pieces:
            if current and current_tokens + piece.tokens > self._chunk_size:
                spans.append((current[0].start, current[-1].end))
                current = self._overlap_tail(current)
                current_tokens = sum(p.tokens for p in current)
                while current and current_tokens + piece.tokens > self._chunk_size:
                    current_tokens -= current.pop(0).tokens
            current.append(piece)
            current_tokens += piece.tokens
        if current:
            spans.append((current[0].start, current[-1].end))
        return spans

    def _overlap_tail(self, pieces: list[_Piece]) -> list[_Piece]:
        """Trailing pieces worth at most ``chunk_overlap`` tokens, to start the next chunk."""
        tail: list[_Piece] = []
        tokens = 0
        for piece in reversed(pieces):
            if tokens + piece.tokens > self._chunk_overlap:
                break
            tail.insert(0, piece)
            tokens += piece.tokens
        return tail

    # ------------------------------------------------------------------ build

    def _make_chunk(self, document: Document, start: int, end: int) -> Chunk:
        chunk_text = document.text[start:end]
        return Chunk(
            chunk_id=make_chunk_id(document.doc_id, start, end, self.fingerprint),
            doc_id=document.doc_id,
            text=chunk_text,
            char_start=start,
            char_end=end,
            page_start=document.page_index_at(start),
            page_end=document.page_index_at(end - 1),
            metadata={
                **document.metadata,
                "chunker": self.fingerprint,
                "token_count": self._counter.count(chunk_text),
            },
        )


def _split_keep_separator(text: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """Spans tiling ``[start, end)``, each ending just after an occurrence of ``separator``."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        found = text.find(separator, cursor, end)
        if found == -1:
            spans.append((cursor, end))
            break
        piece_end = found + len(separator)
        spans.append((cursor, piece_end))
        cursor = piece_end
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
