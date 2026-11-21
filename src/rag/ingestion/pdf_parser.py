"""Baseline PDF parser built on pypdf (study guide, Part 3; HLD 002).

Deliberately naive: raw ``extract_text()`` per page, no layout analysis, no table
handling. Its job is to be the *baseline* that later parsers are measured against
with the parse audit.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

import pypdf
from pypdf.errors import PyPdfError

from rag.domain import Document, Page
from rag.errors import IngestionError
from rag.ingestion.base import Parser

logger = logging.getLogger(__name__)


class PypdfParser(Parser):
    NAME = "pypdf"

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
                text = self._page_text(raw_page)
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

    def _page_text(self, page: pypdf.PageObject) -> str:
        """Raw ``extract_text()``: the naive baseline."""
        return page.extract_text() or ""


_COLUMN_GAP: Final = re.compile(r" {3,}")
"""Layout mode pads table columns with runs of spaces; prose uses single spaces."""
_CURRENCY_GAP: Final = re.compile(r"([$€£¥])(?: \| | )(?=[\d(])")
"""``$ 5,363`` or ``$ | 5,363`` (currency sign in its own column) → ``$5,363``."""
_PERCENT_GAP: Final = re.compile(r"(?<=[\d)])(?: \| | +)%(?! ?[A-Za-z])")
"""``37.2 %`` or ``37.2 | %`` → ``37.2%``: only after a number, and not before a word
(a header like ``2017 | % change`` stays as it is)."""
_BLANK_LINES: Final = re.compile(r"\n{3,}")


def normalize_layout_text(text: str) -> str:
    """Make pypdf's layout-mode text compact while keeping each table row on one line.

    - Non-breaking spaces become spaces.
    - A run of 3+ spaces (a column gap) becomes `` | ``, so a row reads
      ``Purchases of PP&E | (1,577) | (1,373) | (1,420)``.
    - ``$ 5,363`` becomes ``$5,363`` and ``37.2 %`` becomes ``37.2%``, so a value stays
      one cell.
    - Lines are stripped; runs of blank lines collapse to one.

    Hyphens at line ends are kept: in these filings they are mostly real compounds split
    by the layout (``local-`` / ``currency``), not syllable breaks.
    """
    text = text.replace("\xa0", " ")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        line = _COLUMN_GAP.sub(" | ", line)
        line = _CURRENCY_GAP.sub(r"\1", line)
        line = _PERCENT_GAP.sub("%", line)
        lines.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip("\n")


_WORD: Final = re.compile(r"[a-z0-9]+")


def _word_count(text: str) -> int:
    return len(_WORD.findall(text.lower()))


class PypdfLayoutParser(PypdfParser):
    """pypdf's layout mode + :func:`normalize_layout_text` (backlog 3.1; HLD 021).

    The baseline puts every table cell on its own line, so a row label and its values end up
    several lines apart and the column years appear only at the top of the page. Layout mode
    keeps the visual rows, and normalisation turns column gaps into ``|`` separators.

    **Never lose text:** on 31 of 11,948 FinanceBench pages (mostly 10-K cover pages),
    layout mode returned nothing or far fewer words than plain extraction. When layout text
    has fewer than ``min_word_ratio`` of plain extraction's words, the page falls back to
    plain text. Fallbacks are counted in ``metadata["layout_fallback_pages"]``.
    """

    NAME = "pypdf-layout"

    def __init__(self, min_word_ratio: float = 0.8) -> None:
        if not 0 < min_word_ratio <= 1:
            raise ValueError("min_word_ratio must be in (0, 1]")
        self._min_word_ratio = min_word_ratio
        self._fallbacks = 0

    def parse(self, path: Path, doc_id: str) -> Document:
        self._fallbacks = 0
        document = super().parse(path, doc_id)
        return document.model_copy(
            update={"metadata": {**document.metadata, "layout_fallback_pages": self._fallbacks}}
        )

    def _page_text(self, page: pypdf.PageObject) -> str:
        plain = page.extract_text() or ""
        layout = normalize_layout_text(page.extract_text(extraction_mode="layout") or "")
        if _word_count(layout) < self._min_word_ratio * _word_count(plain):
            self._fallbacks += 1
            return plain
        return layout
