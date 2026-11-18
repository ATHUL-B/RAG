"""Baseline PDF parser built on pypdf (study guide, Part 3; HLD 002).

Deliberately naive: raw ``extract_text()`` per page, no layout analysis, no table
handling. Its job is to be the *baseline* that later parsers are measured against
with the parse audit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pypdf
from pypdf.errors import PyPdfError

from rag.domain import Document, Page
from rag.errors import IngestionError
from rag.ingestion.base import Parser

logger = logging.getLogger(__name__)


class PypdfParser(Parser):
    NAME: Final = "pypdf"

    @property
    def name(self) -> str:
        """Parser name + library version: recorded on every document it produces."""
        return f"{self.NAME}-{pypdf.__version__}"

    def parse(self, path: Path, doc_id: str) -> Document:
        try:
            reader = pypdf.PdfReader(path)
            raw_pages = list(reader.pages)
        except (OSError, PyPdfError, ValueError) as exc:
            raise IngestionError(f"cannot open {path}: {exc}") from exc

        pages: list[Page] = []
        failed_pages = 0
        for index, raw_page in enumerate(raw_pages):
            try:
                text = raw_page.extract_text() or ""
            except Exception as exc:  # pypdf raises many exception types on malformed pages
                # Keep an empty page so page indices stay aligned with the source.
                logger.warning(
                    "page extraction failed",
                    extra={"doc_id": doc_id, "page_index": index, "error": repr(exc)},
                )
                text = ""
                failed_pages += 1
            pages.append(Page(page_index=index, text=text))

        return Document(
            doc_id=doc_id,
            source_uri=path.as_posix(),
            pages=tuple(pages),
            metadata={
                "parser": self.name,
                "page_count": len(pages),
                "failed_pages": failed_pages,
            },
        )
