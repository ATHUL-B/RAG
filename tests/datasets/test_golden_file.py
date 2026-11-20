import json
from pathlib import Path

import pytest

from rag.datasets.golden_file import read_golden_items
from rag.errors import DatasetError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(item_id: str, evidence: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {"item_id": item_id, "question": "q?", "reference_answer": "r", "evidence": evidence or []}
    )


def test_reads_items_and_empty_evidence_means_unanswerable(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(_row("a") + "\n\n" + _row("b", [{"doc_id": "D", "page_index": 1}]) + "\n")
    a, b = read_golden_items(path)
    assert not a.answerable and b.answerable


@pytest.mark.parametrize(
    ("content", "message"),
    [("not json\n", "set.jsonl:1"), (_row("a") + "\n" + _row("a") + "\n", "duplicate")],
)
def test_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(content)
    with pytest.raises(DatasetError, match=message):
        read_golden_items(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        read_golden_items(tmp_path / "nope.jsonl")


def test_committed_unanswerable_set_is_valid_and_tagged() -> None:
    items = read_golden_items(REPO_ROOT / "evals" / "unanswerable-v1.jsonl")
    assert len(items) == 26
    assert all(not item.answerable for item in items)
    assert all("type:unanswerable" in item.tags for item in items)
    reasons = {t for item in items for t in item.tags if t.startswith("unanswerable:")}
    assert reasons == {
        "unanswerable:company-in-catalog-no-filings",
        "unanswerable:company-not-in-catalog",
        "unanswerable:period-not-covered",
        "unanswerable:not-disclosed",
        "unanswerable:speculative",
    }
