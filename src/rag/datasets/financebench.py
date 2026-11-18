"""FinanceBench adapter: download the files and map records to domain objects.

Source: https://github.com/patronus-ai/financebench (see docs/adr/0001-dataset.md).

On-disk layout under ``root`` (normally ``data/raw/financebench``)::

    financebench_open_source.jsonl          150 questions with evidence
    financebench_document_information.jsonl 361 filings: company, year, type
    pdfs/<doc_name>.pdf

Upstream facts this module relies on (verified 2026-11-18):
- ``evidence_page_num`` is ZERO-based.
- each evidence item names its document under the key ``doc_name``.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, ValidationError

from rag.datasets.http import Fetcher
from rag.domain import GoldenItem, Metadata, SourceSpan
from rag.errors import DatasetError
from rag.storage import atomic_write_bytes

logger = logging.getLogger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCUMENTS_FILE = "financebench_document_information.jsonl"
PDF_DIR = "pdfs"
_PDF_MAGIC = b"%PDF"
_NO_MAGIC = b""  # metadata files: no content-type check
_SAFE_DOC_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_REASONING_SEPARATOR = re.compile(r"\s+OR\b\s*")


# --------------------------------------------------------------------------- raw records


class _UpstreamRecord(BaseModel):
    """Upstream schema: ignore fields we don't use, but require the ones we do."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class FinanceBenchEvidence(_UpstreamRecord):
    doc_name: str
    evidence_page_num: NonNegativeInt
    evidence_text: str = ""


class FinanceBenchQuestion(_UpstreamRecord):
    financebench_id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None = None
    question: str
    answer: str
    justification: str | None = None
    evidence: list[FinanceBenchEvidence] = Field(default_factory=list)


class FinanceBenchDocument(_UpstreamRecord):
    doc_name: str
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    doc_link: str


def read_jsonl[RecordT: _UpstreamRecord](path: Path, model: type[RecordT]) -> list[RecordT]:
    """Parse a JSONL file into validated records; errors name the file and line."""
    if not path.exists():
        raise DatasetError(f"{path} not found; run scripts/download_financebench.py first")
    records: list[RecordT] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid {model.__name__}: {exc}") from exc
    return records


# --------------------------------------------------------------------------- mapping


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def question_tags(question: FinanceBenchQuestion) -> frozenset[str]:
    """Segment tags used to report metrics per question type (Part 13)."""
    tags = {f"type:{_slug(question.question_type)}"}
    if question.question_reasoning:
        for part in _REASONING_SEPARATOR.split(question.question_reasoning):
            if slug := _slug(part):
                tags.add(f"reasoning:{slug}")
    distinct_pages = {(e.doc_name, e.evidence_page_num) for e in question.evidence}
    tags.add("evidence:multi" if len(distinct_pages) > 1 else "evidence:single")
    return frozenset(tags)


def to_golden_item(question: FinanceBenchQuestion) -> GoldenItem:
    return GoldenItem(
        item_id=question.financebench_id,
        question=question.question,
        reference_answer=question.answer,
        evidence=tuple(
            SourceSpan(doc_id=e.doc_name, page_index=e.evidence_page_num, text=e.evidence_text)
            for e in question.evidence
        ),
        tags=question_tags(question),
        metadata={"company": question.company, "doc_id": question.doc_name},
    )


def document_metadata(document: FinanceBenchDocument) -> Metadata:
    """Filterable metadata for a filing: the fields Part 8's filters will use."""
    return {
        "company": document.company,
        "sector": document.gics_sector,
        "doc_type": document.doc_type,
        "year": document.doc_period,
    }


# --------------------------------------------------------------------------- dataset


class FinanceBenchDataset:
    """Read-only view over a downloaded copy of FinanceBench."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def questions(self) -> list[FinanceBenchQuestion]:
        return read_jsonl(self._root / QUESTIONS_FILE, FinanceBenchQuestion)

    def documents(self) -> list[FinanceBenchDocument]:
        return read_jsonl(self._root / DOCUMENTS_FILE, FinanceBenchDocument)

    def golden_items(self) -> list[GoldenItem]:
        return [to_golden_item(q) for q in self.questions()]

    def referenced_doc_names(self) -> list[str]:
        """Documents that at least one question (or its evidence) points to."""
        names = set[str]()
        for q in self.questions():
            names.add(q.doc_name)
            names.update(e.doc_name for e in q.evidence)
        return sorted(names)

    def pdf_path(self, doc_name: str) -> Path:
        return self._root / PDF_DIR / f"{_validated_doc_name(doc_name)}.pdf"


def _validated_doc_name(doc_name: str) -> str:
    """Doc names come from downloaded data: never let them escape the target dir."""
    if not _SAFE_DOC_NAME.fullmatch(doc_name):
        raise DatasetError(f"unsafe document name: {doc_name!r}")
    return doc_name


# --------------------------------------------------------------------------- download


@dataclass
class DownloadReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class FinanceBenchDownloader:
    """Idempotent downloader: existing files are skipped, writes are atomic."""

    def __init__(self, fetcher: Fetcher, root: Path, base_url: str = RAW_BASE_URL) -> None:
        self._fetcher = fetcher
        self._root = root
        self._base_url = base_url.rstrip("/")

    def download_metadata(self, overwrite: bool = False) -> DownloadReport:
        report = DownloadReport()
        for name in (QUESTIONS_FILE, DOCUMENTS_FILE):
            url = f"{self._base_url}/data/{name}"
            self._download(name, url, self._root / name, report, overwrite, _NO_MAGIC)
        return report

    def download_pdfs(self, doc_names: Iterable[str]) -> DownloadReport:
        """Download each PDF; one failure is recorded and does not stop the rest."""
        report = DownloadReport()
        dataset = FinanceBenchDataset(self._root)
        for doc_name in _unique(doc_names):
            try:
                target = dataset.pdf_path(doc_name)
            except DatasetError as exc:
                report.failed[doc_name] = str(exc)
                continue
            url = f"{self._base_url}/{PDF_DIR}/{urllib.parse.quote(doc_name)}.pdf"
            self._download(doc_name, url, target, report, overwrite=False, magic=_PDF_MAGIC)
        return report

    def _download(
        self,
        name: str,
        url: str,
        target: Path,
        report: DownloadReport,
        overwrite: bool,
        magic: bytes,
    ) -> None:
        if target.exists() and not overwrite:
            report.skipped.append(name)
            return
        try:
            body = self._fetcher.fetch(url)
            if magic and not body.startswith(magic):
                raise DatasetError(f"{url} did not return the expected file type")
        except DatasetError as exc:
            logger.error("download failed", extra={"item": name, "error": str(exc)})
            report.failed[name] = str(exc)
            return
        atomic_write_bytes(target, body)
        report.downloaded.append(name)
        logger.info("downloaded", extra={"item": name, "bytes": len(body)})


def _unique(items: Iterable[str]) -> Iterator[str]:
    seen = set[str]()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item
