import random
from itertools import pairwise

import pytest

from rag.domain import Chunk, Document, Page
from rag.ingestion.chunkers import RecursiveChunker
from support.tokenizers import CharCounter, WordCounter


def _doc(*pages: str, **metadata: str | int) -> Document:
    return Document(
        doc_id="DOC_2024",
        source_uri="u",
        pages=tuple(Page(page_index=i, text=t) for i, t in enumerate(pages)),
        metadata=metadata,
    )


def _words(chunk: Chunk) -> list[str]:
    return chunk.text.split()


def _assert_offsets_exact(document: Document, chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.text == chunk.text.strip()


class TestSplitting:
    def test_small_document_is_one_chunk(self) -> None:
        doc = _doc("  hello world  ")
        chunks = RecursiveChunker(WordCounter(), chunk_size=10).chunk(doc)
        assert [c.text for c in chunks] == ["hello world"]
        _assert_offsets_exact(doc, chunks)

    def test_prefers_paragraph_boundaries(self) -> None:
        doc = _doc("a b c\n\nd e f\n\ng h")
        chunks = RecursiveChunker(WordCounter(), chunk_size=3).chunk(doc)
        assert [c.text for c in chunks] == ["a b c", "d e f", "g h"]

    def test_packs_small_paragraphs_together(self) -> None:
        doc = _doc("a\n\nb\n\nc\n\nd e f")
        chunks = RecursiveChunker(WordCounter(), chunk_size=3).chunk(doc)
        assert [c.text for c in chunks] == ["a\n\nb\n\nc", "d e f"]

    def test_falls_back_to_lines_then_words(self) -> None:
        doc = _doc("one two three four\nfive six")
        chunks = RecursiveChunker(WordCounter(), chunk_size=2).chunk(doc)
        assert [c.text for c in chunks] == ["one two", "three four", "five six"]

    def test_hard_splits_unbroken_text(self) -> None:
        doc = _doc("abcdefghij")
        chunks = RecursiveChunker(CharCounter(), chunk_size=4).chunk(doc)
        assert "".join(c.text for c in chunks) == "abcdefghij"
        assert all(len(c.text) <= 4 for c in chunks)
        _assert_offsets_exact(doc, chunks)

    def test_whitespace_only_document_has_no_chunks(self) -> None:
        assert RecursiveChunker(WordCounter(), chunk_size=5).chunk(_doc("   ", "\n\n")) == []

    def test_empty_document_has_no_chunks(self) -> None:
        assert RecursiveChunker(WordCounter(), chunk_size=5).chunk(_doc()) == []


class TestOverlap:
    def test_next_chunk_starts_with_tail_of_previous(self) -> None:
        doc = _doc(" ".join(f"w{i}" for i in range(10)))
        chunks = RecursiveChunker(WordCounter(), chunk_size=4, chunk_overlap=2).chunk(doc)
        texts = [_words(c) for c in chunks]
        assert texts[0] == ["w0", "w1", "w2", "w3"]
        for previous, current in pairwise(texts):
            assert current[:2] == previous[-2:]
        assert texts[-1][-1] == "w9"

    def test_no_overlap_means_disjoint_chunks(self) -> None:
        doc = _doc(" ".join(f"w{i}" for i in range(10)))
        chunks = RecursiveChunker(WordCounter(), chunk_size=4).chunk(doc)
        for previous, current in pairwise(chunks):
            assert previous.char_end <= current.char_start


class TestChunkRecords:
    def test_spans_pages_and_copies_metadata(self) -> None:
        doc = _doc("alpha beta", "gamma delta", company="ACME", year=2024)
        (chunk,) = RecursiveChunker(WordCounter(), chunk_size=10).chunk(doc)
        assert (chunk.page_start, chunk.page_end) == (0, 1)
        assert chunk.metadata["company"] == "ACME"
        assert chunk.metadata["year"] == 2024
        assert chunk.metadata["token_count"] == 4
        assert chunk.metadata["chunker"] == RecursiveChunker(WordCounter(), 10).fingerprint

    def test_page_of_each_chunk(self) -> None:
        doc = _doc("a b c", "d e f")
        chunks = RecursiveChunker(WordCounter(), chunk_size=3).chunk(doc)
        assert [(c.page_start, c.page_end) for c in chunks] == [(0, 0), (1, 1)]

    def test_ids_are_deterministic_and_unique(self) -> None:
        doc = _doc("a b c\n\nd e f\n\ng h")
        first = RecursiveChunker(WordCounter(), chunk_size=3).chunk(doc)
        second = RecursiveChunker(WordCounter(), chunk_size=3).chunk(doc)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert len({c.chunk_id for c in first}) == len(first)


class TestFingerprint:
    @pytest.mark.parametrize(
        "other",
        [
            RecursiveChunker(WordCounter(), chunk_size=401),
            RecursiveChunker(WordCounter(), chunk_size=400, chunk_overlap=40),
            RecursiveChunker(CharCounter(), chunk_size=400),
            RecursiveChunker(WordCounter(), chunk_size=400, separators=["\n"]),
        ],
    )
    def test_changes_with_any_setting(self, other: RecursiveChunker) -> None:
        assert RecursiveChunker(WordCounter(), chunk_size=400).fingerprint != other.fingerprint

    @pytest.mark.parametrize(
        ("size", "overlap", "separators"),
        [(0, 0, ["\n"]), (10, 10, ["\n"]), (10, -1, ["\n"]), (10, 0, ["\n", ""])],
    )
    def test_rejects_invalid_settings(self, size: int, overlap: int, separators: list[str]) -> None:
        with pytest.raises(ValueError):
            RecursiveChunker(WordCounter(), size, overlap, separators)


@pytest.mark.parametrize("seed", range(5))
def test_random_documents_keep_invariants(seed: int) -> None:
    """Property check: exact offsets, size bound, ordering, and every word kept."""
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "x", "Revenue", "1,577", "(in", "millions)"]
    separators = [" ", " ", " ", "\n", "\n\n", ". "]
    pages = [
        "".join(rng.choice(vocab) + rng.choice(separators) for _ in range(rng.randint(0, 120)))
        for _ in range(rng.randint(1, 4))
    ]
    doc = _doc(*pages)
    chunks = RecursiveChunker(WordCounter(), chunk_size=25).chunk(doc)

    _assert_offsets_exact(doc, chunks)
    assert all(WordCounter().count(c.text) <= 25 for c in chunks)
    assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)
    assert [w for c in chunks for w in _words(c)] == doc.text.split()
