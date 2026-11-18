"""Metadata enrichment (study guide, Part 3 "Metadata enrichment", used by Part 8 filters)."""

from __future__ import annotations

from collections.abc import Mapping

from rag.domain import Document, MetadataValue
from rag.errors import IngestionError
from rag.ingestion.base import MetadataEnricher


class StaticMetadataEnricher(MetadataEnricher):
    """Adds metadata looked up by ``doc_id`` from a known catalog.

    Missing documents are an error, not a skip: a document without its ``year`` or
    ``company`` would silently escape every metadata filter later.
    """

    def __init__(self, catalog: Mapping[str, Mapping[str, MetadataValue]]) -> None:
        self._catalog = dict(catalog)

    def enrich(self, document: Document) -> Document:
        try:
            extra = self._catalog[document.doc_id]
        except KeyError as exc:
            raise IngestionError(f"no catalog metadata for {document.doc_id!r}") from exc
        clashes = {k for k in extra if k in document.metadata and document.metadata[k] != extra[k]}
        if clashes:
            raise IngestionError(f"metadata conflict for {document.doc_id!r}: {sorted(clashes)}")
        return document.model_copy(update={"metadata": {**document.metadata, **extra}})
