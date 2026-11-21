"""Build tiny but real PDFs in memory, so parser tests exercise pypdf for real."""

from collections.abc import Callable

import pytest

from support.pdfs import build_pdf

MakePdf = Callable[[list[str]], bytes]


@pytest.fixture
def make_pdf() -> MakePdf:
    return build_pdf
