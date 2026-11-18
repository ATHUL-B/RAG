"""Offline ingestion contracts (study guide, Parts 3-4).

Each stage is a small interface with one job. Concrete classes (a PDF parser, a
recursive chunker, ...) implement them and are selected by config, so an
ablation is "swap one implementation, rerun the eval".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from rag.domain import Chunk, Document


class Parser(ABC):
    """Turns a raw file into a :class:`Document`, preserving page boundaries."""

    @abstractmethod
    def parse(self, path: Path, doc_id: str) -> Document: ...


class Cleaner(ABC):
    """Removes noise (headers, footers, hyphenation) without losing meaning."""

    @abstractmethod
    def clean(self, document: Document) -> Document: ...


class MetadataEnricher(ABC):
    """Attaches metadata used later for filtering (company, year, doc type...)."""

    @abstractmethod
    def enrich(self, document: Document) -> Document: ...


class Chunker(ABC):
    """Splits a document into retrievable units with exact source offsets."""

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Hash of this chunker's config; part of every chunk ID it produces."""

    @abstractmethod
    def chunk(self, document: Document) -> Sequence[Chunk]: ...
