import io
from collections.abc import Callable
from pathlib import Path

import pypdf
import pytest

from rag.errors import IngestionError
from rag.ingestion.pdf_parser import PypdfParser

MakePdf = Callable[[list[str]], bytes]


def _write(tmp_path: Path, body: bytes, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _encrypt(body: bytes, user_password: str) -> bytes:
    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(body)))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_extracts_one_page_per_pdf_page_in_order(tmp_path: Path, make_pdf: MakePdf) -> None:
    path = _write(tmp_path, make_pdf(["Revenue 100", "Capex (1,577)", "Notes"]))
    document = PypdfParser().parse(path, "ACME_2024_10K")

    assert document.doc_id == "ACME_2024_10K"
    assert [p.page_index for p in document.pages] == [0, 1, 2]
    assert "Capex (1,577)" in document.pages[1].text
    assert document.metadata["page_count"] == 3
    assert document.metadata["failed_pages"] == 0
    assert str(document.metadata["parser"]).startswith("pypdf-")
    assert document.source_uri == path.as_posix()


def test_reads_owner_locked_aes_pdf_with_empty_user_password(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    """Several FinanceBench filings are encrypted like this; needs pypdf[crypto]."""
    path = _write(tmp_path, _encrypt(make_pdf(["secret revenue"]), user_password=""))
    assert "secret revenue" in PypdfParser().parse(path, "d").pages[0].text


def test_password_protected_pdf_raises_ingestion_error(tmp_path: Path, make_pdf: MakePdf) -> None:
    path = _write(tmp_path, _encrypt(make_pdf(["x"]), user_password="hunter2"))
    with pytest.raises(IngestionError, match="cannot open"):
        PypdfParser().parse(path, "d")


@pytest.mark.parametrize("body", [b"not a pdf at all", b""])
def test_corrupt_file_raises_ingestion_error(tmp_path: Path, body: bytes) -> None:
    with pytest.raises(IngestionError, match="cannot open"):
        PypdfParser().parse(_write(tmp_path, body), "d")


def test_missing_file_raises_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="cannot open"):
        PypdfParser().parse(tmp_path / "nope.pdf", "d")


def test_failed_page_becomes_empty_and_keeps_alignment(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pypdf.PageObject.extract_text

    def flaky(self: pypdf.PageObject, *args: object, **kwargs: object) -> str:
        text = original(self)
        if "BAD" in text:
            raise KeyError("/Font")
        return text

    monkeypatch.setattr(pypdf.PageObject, "extract_text", flaky)
    path = _write(tmp_path, make_pdf(["good one", "BAD page", "good two"]))
    document = PypdfParser().parse(path, "d")

    assert [p.text for p in document.pages][1] == ""
    assert "good two" in document.pages[2].text
    assert document.metadata["failed_pages"] == 1


# --- layout parser (backlog 3.1, HLD 021) ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Purchases of PP&E        (1,577)        (1,373)       (1,420)",
            "Purchases of PP&E | (1,577) | (1,373) | (1,420)",
        ),
        ("Net income      $      5,363     $    4,869", "Net income | $5,363 | $4,869"),
        ("Industrial   $    2,952      37.2   %", "Industrial | $2,952 | 37.2%"),
        ("Margin   8.1  %", "Margin | 8.1%"),
        ("December 31, 2017      % change", "December 31, 2017 | % change"),  # header stays
        ("Net    % of", "Net | % of"),
        ("Other\xa0\xa0\xa0net", "Other | net"),
        ("Organic local-\ncurrency sales", "Organic local-\ncurrency sales"),  # compound kept
        ("One sentence with single spaces.", "One sentence with single spaces."),
        ("   Title\n\n\n\n\nBody   ", "Title\n\nBody"),
        ("", ""),
    ],
)
def test_normalize_layout_text(raw: str, expected: str) -> None:
    from rag.ingestion.pdf_parser import normalize_layout_text

    assert normalize_layout_text(raw) == expected


def test_layout_parser_keeps_a_table_row_on_one_line(tmp_path: Path) -> None:
    from rag.ingestion.pdf_parser import PypdfLayoutParser
    from support.pdfs import build_positioned_pdf

    page = [
        (72, 720, "(Millions)"),
        (330, 720, "2018"),
        (430, 720, "2017"),
        (72, 700, "Purchases of PP&E"),
        (330, 700, "(1,577)"),
        (430, 700, "(1,373)"),
    ]
    path = _write(tmp_path, build_positioned_pdf([page]))
    document = PypdfLayoutParser().parse(path, "3M_2018_10K")
    lines = document.pages[0].text.splitlines()
    assert "(Millions) | 2018 | 2017" in lines
    assert "Purchases of PP&E | (1,577) | (1,373)" in lines
    assert str(document.metadata["parser"]).startswith("pypdf-layout-")


def test_layout_parser_falls_back_to_plain_text_when_layout_loses_words(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag.ingestion.pdf_parser import PypdfLayoutParser

    original = pypdf.PageObject.extract_text

    def layout_loses_page_two(self: pypdf.PageObject, *args: object, **kwargs: object) -> str:
        text = original(self)
        if kwargs.get("extraction_mode") == "layout" and "cover" in text:
            return ""  # what pypdf's layout mode did on 14 FinanceBench cover pages
        return text

    monkeypatch.setattr(pypdf.PageObject, "extract_text", layout_loses_page_two)
    path = _write(tmp_path, make_pdf(["Revenue 100", "cover page text"]))
    document = PypdfLayoutParser().parse(path, "d")
    assert document.pages[1].text == "cover page text"
    assert document.metadata["layout_fallback_pages"] == 1


def test_layout_parser_rejects_bad_ratio() -> None:
    from rag.ingestion.pdf_parser import PypdfLayoutParser

    with pytest.raises(ValueError, match="min_word_ratio"):
        PypdfLayoutParser(min_word_ratio=0)
