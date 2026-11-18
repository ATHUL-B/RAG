import pytest

from rag.domain import Chunk, Document, Page
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.context import (
    ContextualChunker,
    HeaderContextualizer,
    page_title_block,
)
from support.tokenizers import WordCounter

STATEMENT_PAGE = """Table of Contents
3M Company and Subsidiaries
Consolidated Statement of Cash Flow
s
Years ended December 31
(Millions)
2018
2017
2016
Cash Flows from Operating Activities
Net income including noncontrolling interest"""

PROSE_PAGE = (
    "partially offset by the decreases in Asia Pacific. Selling prices were flat in 2017 and\n"
    "Financial condition:"
)


class TestPageTitleBlock:
    def test_statement_page_gives_entity_title_period_and_years(self) -> None:
        assert page_title_block(STATEMENT_PAGE) == (
            "3M Company and Subsidiaries / Consolidated Statement of Cash Flows / "
            "Years ended December 31 / (Millions) / 2018 / 2017 / 2016 / "
            "Cash Flows from Operating Activities"
        )  # the re-joined fragment "s" does not use up one of the 8 lines

    def test_prose_page_has_no_title(self) -> None:
        assert page_title_block(PROSE_PAGE) == ""

    def test_broken_table_of_contents_and_page_numbers_are_dropped(self) -> None:
        text = "T\nable of\nContents\n128\nNote 5 - Acquisitions\nYear ended June 30, 2023"
        assert page_title_block(text) == "Note 5 - Acquisitions / Year ended June 30, 2023"

    def test_limits_lines_and_characters(self) -> None:
        text = "\n".join(f"Line {i}" for i in range(20))
        assert page_title_block(text, max_lines=2) == "Line 0 / Line 1"
        assert len(page_title_block(text, max_chars=10)) <= 10

    def test_non_breaking_spaces_are_normalized(self) -> None:
        assert page_title_block("Balance\xa0Sheets\nAt\xa0December 31") == (
            "Balance Sheets / At December 31"
        )


def _document() -> Document:
    return Document(
        doc_id="3M_2018_10K",
        source_uri="u",
        pages=(Page(page_index=0, text=PROSE_PAGE), Page(page_index=1, text=STATEMENT_PAGE)),
        metadata={"company": "3M", "doc_type": "10k", "year": 2018},
    )


class TestHeaderContextualizer:
    def test_combines_document_fields_and_page_title(self) -> None:
        document = _document()
        chunks = RecursiveChunker(WordCounter(), 50, respect_pages=True).chunk(document)
        contexts = HeaderContextualizer().contextualize(document, chunks)
        assert contexts[0] == "3M | 10k | 2018"  # prose page: no title block
        assert contexts[-1].startswith("3M | 10k | 2018 | 3M Company and Subsidiaries / ")

    def test_missing_fields_and_disabled_title(self) -> None:
        document = _document()
        chunks = RecursiveChunker(WordCounter(), 50, respect_pages=True).chunk(document)
        contexts = HeaderContextualizer(("company", "sector"), page_title=False).contextualize(
            document, chunks
        )
        assert set(contexts) == {"3M"}

    def test_rejects_invalid_title_lines(self) -> None:
        with pytest.raises(ValueError, match="max_title_lines"):
            HeaderContextualizer(max_title_lines=0)


class TestContextualChunker:
    def test_same_boundaries_new_ids_and_indexed_text(self) -> None:
        document = _document()
        inner = RecursiveChunker(WordCounter(), 50, respect_pages=True)
        outer = ContextualChunker(inner, HeaderContextualizer())
        plain, contextual = inner.chunk(document), outer.chunk(document)

        assert [(c.char_start, c.char_end) for c in plain] == [
            (c.char_start, c.char_end) for c in contextual
        ]
        assert {c.chunk_id for c in plain}.isdisjoint({c.chunk_id for c in contextual})
        assert outer.fingerprint not in {inner.fingerprint, HeaderContextualizer().fingerprint}
        assert all(c.metadata["chunker"] == outer.fingerprint for c in contextual)
        last = contextual[-1]
        assert last.text == document.text[last.char_start : last.char_end]
        assert last.indexed_text == f"{last.context}\n\n{last.text}"

    def test_fingerprint_depends_on_contextualizer_settings(self) -> None:
        inner = RecursiveChunker(WordCounter(), 50)
        a = ContextualChunker(inner, HeaderContextualizer())
        b = ContextualChunker(inner, HeaderContextualizer(page_title=False))
        assert a.fingerprint != b.fingerprint


def test_chunk_without_context_indexes_its_text() -> None:
    chunk = Chunk(
        chunk_id="c", doc_id="d", text="abc", char_start=0, char_end=3, page_start=0, page_end=0
    )
    assert chunk.indexed_text == "abc"
