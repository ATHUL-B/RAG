import json
from pathlib import Path
from typing import Any

import pytest

from rag.datasets.financebench import (
    DOCUMENTS_FILE,
    QUESTIONS_FILE,
    FinanceBenchDataset,
    FinanceBenchDocument,
    FinanceBenchDownloader,
    FinanceBenchQuestion,
    document_metadata,
    question_tags,
    to_golden_item,
)
from rag.datasets.http import Fetcher
from rag.domain import SourceSpan
from rag.errors import DatasetError

BASE = "https://fake.test/repo"
PDF_BYTES = b"%PDF-1.4 fake"


def _question(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "financebench_id": "fb_1",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question_type": "metrics-generated",
        "question_reasoning": "Numerical reasoning OR Logical reasoning OR",
        "question": "What was capex in FY2018?",
        "answer": "$1577.00",
        "justification": "from cash flow statement",
        "dataset_subset_label": "OPEN_SOURCE",  # unused upstream field: ignored
        "evidence": [
            {
                "doc_name": "3M_2018_10K",
                "evidence_page_num": 59,
                "evidence_text": "Purchases of PP&E (1,577)",
                "evidence_text_full_page": "...",
            }
        ],
    }
    return record | overrides


_DOCUMENT = {
    "doc_name": "3M_2018_10K",
    "company": "3M",
    "gics_sector": "Industrials",
    "doc_type": "10k",
    "doc_period": 2018,
    "doc_link": "https://example.test/3m.pdf",
}


class FakeFetcher(Fetcher):
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.responses:
            raise DatasetError(f"HTTP 404 fetching {url}")
        return self.responses[url]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> FinanceBenchDataset:
    second = _question(
        financebench_id="fb_2",
        doc_name="AMD_2022_10K",
        question_reasoning=None,
        evidence=[
            {"doc_name": "AMD_2022_10K", "evidence_page_num": 3},
            {"doc_name": "AMD_2021_10K", "evidence_page_num": 7},
        ],
    )
    _write_jsonl(tmp_path / QUESTIONS_FILE, [_question(), second])
    _write_jsonl(tmp_path / DOCUMENTS_FILE, [_DOCUMENT])
    return FinanceBenchDataset(tmp_path)


class TestMapping:
    def test_to_golden_item_keeps_zero_based_page_and_evidence(self) -> None:
        item = to_golden_item(FinanceBenchQuestion.model_validate(_question()))
        assert item.item_id == "fb_1"
        assert item.reference_answer == "$1577.00"
        assert item.evidence == (
            SourceSpan(doc_id="3M_2018_10K", page_index=59, text="Purchases of PP&E (1,577)"),
        )
        assert item.metadata == {"company": "3M", "doc_id": "3M_2018_10K"}

    def test_tags_split_reasoning_and_drop_dangling_or(self) -> None:
        tags = question_tags(FinanceBenchQuestion.model_validate(_question()))
        assert tags == {
            "type:metrics-generated",
            "reasoning:numerical-reasoning",
            "reasoning:logical-reasoning",
            "evidence:single",
        }

    def test_multi_page_evidence_is_tagged(self) -> None:
        evidence = [
            {"doc_name": "A", "evidence_page_num": 1},
            {"doc_name": "A", "evidence_page_num": 2},
        ]
        tags = question_tags(FinanceBenchQuestion.model_validate(_question(evidence=evidence)))
        assert "evidence:multi" in tags

    def test_document_metadata_exposes_filter_fields(self) -> None:
        meta = document_metadata(FinanceBenchDocument.model_validate(_DOCUMENT))
        assert meta == {"company": "3M", "sector": "Industrials", "doc_type": "10k", "year": 2018}


class TestDataset:
    def test_reads_questions_and_documents(self, dataset: FinanceBenchDataset) -> None:
        assert [q.financebench_id for q in dataset.questions()] == ["fb_1", "fb_2"]
        assert dataset.documents()[0].doc_period == 2018
        assert len(dataset.golden_items()) == 2

    def test_referenced_docs_include_evidence_docs(self, dataset: FinanceBenchDataset) -> None:
        assert dataset.referenced_doc_names() == ["3M_2018_10K", "AMD_2021_10K", "AMD_2022_10K"]

    def test_missing_file_explains_how_to_download(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="download_financebench"):
            FinanceBenchDataset(tmp_path).questions()

    def test_malformed_line_reports_file_and_line(self, tmp_path: Path) -> None:
        (tmp_path / QUESTIONS_FILE).write_text('{"financebench_id": "x"}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match=rf"{QUESTIONS_FILE}:1"):
            FinanceBenchDataset(tmp_path).questions()

    @pytest.mark.parametrize("name", ["../etc/passwd", "a/b", "", "x.pdf"])
    def test_pdf_path_rejects_unsafe_names(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(DatasetError, match="unsafe"):
            FinanceBenchDataset(tmp_path).pdf_path(name)


class TestDownloader:
    def test_downloads_metadata(self, tmp_path: Path) -> None:
        fetcher = FakeFetcher(
            {f"{BASE}/data/{QUESTIONS_FILE}": b"q", f"{BASE}/data/{DOCUMENTS_FILE}": b"d"}
        )
        report = FinanceBenchDownloader(fetcher, tmp_path, BASE).download_metadata()
        assert report.downloaded == [QUESTIONS_FILE, DOCUMENTS_FILE]
        assert (tmp_path / QUESTIONS_FILE).read_bytes() == b"q"

    def test_pdfs_are_idempotent_and_deduplicated(self, tmp_path: Path) -> None:
        fetcher = FakeFetcher({f"{BASE}/pdfs/A_2020_10K.pdf": PDF_BYTES})
        downloader = FinanceBenchDownloader(fetcher, tmp_path, BASE)

        first = downloader.download_pdfs(["A_2020_10K", "A_2020_10K"])
        second = downloader.download_pdfs(["A_2020_10K"])

        assert first.downloaded == ["A_2020_10K"]
        assert second.skipped == ["A_2020_10K"]
        assert len(fetcher.calls) == 1
        assert (tmp_path / "pdfs" / "A_2020_10K.pdf").read_bytes() == PDF_BYTES
        assert not list(tmp_path.rglob("*.part"))

    def test_one_failure_does_not_stop_the_rest(self, tmp_path: Path) -> None:
        fetcher = FakeFetcher(
            {f"{BASE}/pdfs/OK_2020_10K.pdf": PDF_BYTES, f"{BASE}/pdfs/HTML_2020_10K.pdf": b"<html"}
        )
        report = FinanceBenchDownloader(fetcher, tmp_path, BASE).download_pdfs(
            ["MISSING_2020_10K", "../escape", "HTML_2020_10K", "OK_2020_10K"]
        )
        assert report.downloaded == ["OK_2020_10K"]
        assert set(report.failed) == {"MISSING_2020_10K", "../escape", "HTML_2020_10K"}
        assert "expected file type" in report.failed["HTML_2020_10K"]
        assert not (tmp_path / "pdfs" / "HTML_2020_10K.pdf").exists()
