"""Plain data objects passed between pipeline stages.

All models are immutable (``frozen=True``): a stage never edits its input, it
returns a new object. That keeps every intermediate artifact trustworthy when we
write it to disk and later ask "what exactly did the indexer see?".

Conventions
-----------
- Page numbers are **zero-based** (``page_index``), matching PDF libraries and
  FinanceBench's ``evidence_page_num``.
- Character offsets are half-open ``[char_start, char_end)`` into
  :attr:`Document.text`.
"""

from __future__ import annotations

from functools import cached_property
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

FilterValue = MetadataValue | tuple[MetadataValue, ...]
"""A scalar means "equals"; a tuple means "any of" (e.g. ``year: (2022, 2023)``)."""
Filters = dict[str, FilterValue]


def matches(metadata: Metadata, filters: Filters) -> bool:
    """True if ``metadata`` satisfies every filter (the one definition used everywhere)."""
    for key, wanted in filters.items():
        if key not in metadata:
            return False
        value = metadata[key]
        if isinstance(wanted, tuple):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


PAGE_SEPARATOR = "\n\n"
"""Inserted between pages when building :attr:`Document.text`."""


class DomainModel(BaseModel):
    """Base for all domain objects: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Page(DomainModel):
    page_index: NonNegativeInt
    text: str


class Document(DomainModel):
    """A parsed source document, kept page by page so evidence can be located."""

    doc_id: str = Field(min_length=1)
    source_uri: str
    pages: tuple[Page, ...]
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _pages_are_ordered(self) -> Self:
        indices = [p.page_index for p in self.pages]
        if indices != sorted(set(indices)):
            raise ValueError(f"pages of {self.doc_id!r} must be unique and ascending: {indices}")
        return self

    @cached_property
    def text(self) -> str:
        """Full text: pages joined by :data:`PAGE_SEPARATOR`."""
        return PAGE_SEPARATOR.join(page.text for page in self.pages)

    @cached_property
    def _page_starts(self) -> tuple[int, ...]:
        starts, offset = [], 0
        for page in self.pages:
            starts.append(offset)
            offset += len(page.text) + len(PAGE_SEPARATOR)
        return tuple(starts)

    def page_index_at(self, char_offset: int) -> int:
        """Return the ``page_index`` containing ``char_offset`` in :attr:`text`.

        Offsets that fall on a page separator are attributed to the preceding page.
        """
        if not self.pages:
            raise ValueError(f"document {self.doc_id!r} has no pages")
        if not 0 <= char_offset < max(len(self.text), 1):
            raise ValueError(f"offset {char_offset} outside document of length {len(self.text)}")
        position = 0
        for i, start in enumerate(self._page_starts):
            if start > char_offset:
                break
            position = i
        return self.pages[position].page_index


class Chunk(DomainModel):
    """The unit that is indexed, retrieved and cited (study guide, Part 4)."""

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    page_start: NonNegativeInt
    page_end: NonNegativeInt
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spans_are_valid(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class ChunkSet(DomainModel):
    """All chunks of one document produced by one chunker configuration (an artifact)."""

    doc_id: str = Field(min_length=1)
    chunker_fingerprint: str = Field(min_length=1)
    chunks: tuple[Chunk, ...]

    @model_validator(mode="after")
    def _chunks_belong_to_document(self) -> Self:
        strangers = {c.doc_id for c in self.chunks} - {self.doc_id}
        if strangers:
            raise ValueError(f"chunks from other documents in {self.doc_id!r}: {strangers}")
        return self


class SourceSpan(DomainModel):
    """Where a piece of evidence lives in the *source*, independent of chunking.

    Golden labels point here rather than at chunk IDs so they survive re-chunking
    (CLAUDE.md §2).
    """

    doc_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    text: str = ""


class ScoredChunk(DomainModel):
    """A retrieval candidate with the score and rank a specific stage gave it."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    stage: str = Field(min_length=1, description="e.g. 'dense', 'bm25', 'rrf', 'rerank'")


class Query(DomainModel):
    text: str = Field(min_length=1)
    filters: Filters = Field(default_factory=dict)


class Answer(DomainModel):
    text: str
    cited_chunk_ids: tuple[str, ...] = ()
    refused: bool = False


class GoldenItem(DomainModel):
    """One evaluation question with its known evidence and reference answer."""

    item_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str
    evidence: tuple[SourceSpan, ...]
    tags: frozenset[str] = frozenset()
    metadata: Metadata = Field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)
