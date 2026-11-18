"""Contextual chunk headers (study guide, Part 4 "heading path prepended"; HLD 009).

A chunk cut from the middle of a financial statement ("Acquisitions, net of cash acquired
13 (2,023) …") is meaningless on its own. Prepending *where it came from*, i.e. the
filing and the page's title block, gives the embedder, BM25 and rerankers something to
match "FY2018 capital expenditure" against.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property

from rag.domain import Chunk, Document, config_fingerprint, make_chunk_id
from rag.ingestion.base import Chunker

_TABLE_OF_CONTENTS = re.compile(
    r"t\s*a\s*b\s*l\s*e\s+o\s*f\s+c\s*o\s*n\s*t\s*e\s*n\s*t\s*s", re.IGNORECASE
)
_PAGE_NUMBER = re.compile(r"\d{1,3}")
_WORD_FRAGMENT = re.compile(r"[a-z]{1,2}")
_SEPARATOR = " / "


def page_title_block(
    page_text: str, max_lines: int = 8, max_line_chars: int = 60, max_chars: int = 200
) -> str:
    """The short lines at the top of a page (statement/table/note titles), joined.

    Stops at the first long line (prose), drops "Table of Contents" (even when broken
    across lines) and bare page numbers, and re-attaches 1-2 letter fragments that PDF
    extraction split off a word ("Flow" + "s" → "Flows"). Returns "" for prose pages.
    """
    cleaned = _TABLE_OF_CONTENTS.sub(" ", page_text.replace("\xa0", " "))
    lines: list[str] = []
    for raw in cleaned.split("\n"):
        line = " ".join(raw.split())
        if not line or _PAGE_NUMBER.fullmatch(line):
            continue
        if len(line) > max_line_chars:
            break
        if lines and _WORD_FRAGMENT.fullmatch(line):
            lines[-1] += line
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return _SEPARATOR.join(lines)[:max_chars].strip()


class ChunkContextualizer(ABC):
    @property
    @abstractmethod
    def fingerprint(self) -> str: ...

    @abstractmethod
    def contextualize(self, document: Document, chunks: Sequence[Chunk]) -> list[str]:
        """One context string per chunk, in order."""


class HeaderContextualizer(ChunkContextualizer):
    """``<document fields> | <title block of the chunk's first page>``."""

    def __init__(
        self,
        document_fields: Sequence[str] = ("company", "doc_type", "year"),
        page_title: bool = True,
        max_title_lines: int = 8,
    ) -> None:
        if max_title_lines < 1:
            raise ValueError("max_title_lines must be >= 1")
        self._document_fields = tuple(document_fields)
        self._page_title = page_title
        self._max_title_lines = max_title_lines

    @cached_property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {
                "contextualizer": "header",
                "document_fields": list(self._document_fields),
                "page_title": self._page_title,
                "max_title_lines": self._max_title_lines,
            }
        )

    def contextualize(self, document: Document, chunks: Sequence[Chunk]) -> list[str]:
        head = [str(document.metadata[f]) for f in self._document_fields if f in document.metadata]
        titles: dict[int, str] = {}
        if self._page_title:
            texts = {page.page_index: page.text for page in document.pages}
            for page_index in {c.page_start for c in chunks}:
                titles[page_index] = page_title_block(
                    texts.get(page_index, ""), max_lines=self._max_title_lines
                )
        return [" | ".join(p for p in (*head, titles.get(c.page_start, "")) if p) for c in chunks]


class ContextualChunker(Chunker):
    """Decorates any chunker: same boundaries, plus a context header on every chunk.

    Chunk IDs and the fingerprint are re-derived from both parts, so contextual chunks
    land in their own chunk directory and index and never mix with plain ones.
    """

    def __init__(self, inner: Chunker, contextualizer: ChunkContextualizer) -> None:
        self._inner = inner
        self._contextualizer = contextualizer

    @cached_property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {"inner": self._inner.fingerprint, "context": self._contextualizer.fingerprint}
        )

    def chunk(self, document: Document) -> list[Chunk]:
        chunks = list(self._inner.chunk(document))
        contexts = self._contextualizer.contextualize(document, chunks)
        return [
            chunk.model_copy(
                update={
                    "chunk_id": make_chunk_id(
                        chunk.doc_id, chunk.char_start, chunk.char_end, self.fingerprint
                    ),
                    "context": context,
                    "metadata": {**chunk.metadata, "chunker": self.fingerprint},
                }
            )
            for chunk, context in zip(chunks, contexts, strict=True)
        ]
