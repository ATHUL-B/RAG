import json
from pathlib import Path

import pytest

from rag.datasets.beir import MTEB_URL, BeirDataset, download_beir
from rag.datasets.http import Fetcher
from rag.errors import DatasetError


def _write_dataset(root: Path) -> Path:
    d = root / "tiny"
    (d / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Aspirin", "text": "reduces fever."},
        {"_id": "d2", "title": "", "text": "No title here."},
        {"_id": "d3", "title": "", "text": ""},  # FiQA has 38 of these
    ]
    (d / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus))
    queries = [{"_id": "q1", "text": "does aspirin reduce fever"}, {"_id": "q9", "text": "x"}]
    (d / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in queries))
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n")
    return d


def test_corpus_queries_and_graded_qrels(tmp_path: Path) -> None:
    dataset = BeirDataset(_write_dataset(tmp_path))
    chunks = dataset.corpus()
    assert [c.chunk_id for c in chunks] == ["d1", "d2", "d3"]
    assert chunks[2].text == " "  # empty document kept, embeds like MTEB's ""
    assert chunks[0].indexed_text == "Aspirin reduces fever."  # title + " " + text, as MTEB
    assert chunks[1].text == "No title here."  # no leading space without a title
    assert dataset.qrels() == {"q1": {"d1": 2, "d2": 0}}
    assert dataset.queries() == {"q1": "does aspirin reduce fever"}  # only judged queries


def test_missing_files_and_malformed_qrels(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="download_beir"):
        BeirDataset(tmp_path / "nope").corpus()
    d = _write_dataset(tmp_path)
    (d / "qrels" / "test.tsv").write_text("q1\td1\n")
    with pytest.raises(DatasetError, match="3 tab-separated"):
        BeirDataset(d).qrels()


class FakeFetcher(Fetcher):
    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        return url.rsplit("/", 1)[-1].encode()


def test_download_fetches_three_pinned_files_once(tmp_path: Path) -> None:
    fetcher = FakeFetcher()
    target = download_beir("tiny", tmp_path, fetcher, revision="abc123")
    assert fetcher.urls == [
        MTEB_URL.format(name="tiny", revision="abc123", path=p)
        for p in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    ]
    assert (target / "qrels" / "test.tsv").read_bytes() == b"test.tsv"
    download_beir("tiny", tmp_path, fetcher, revision="abc123")  # present: no new fetches
    assert len(fetcher.urls) == 3


@pytest.mark.parametrize("revision", ["", "main/../x", "a/b"])
def test_download_rejects_bad_revisions(tmp_path: Path, revision: str) -> None:
    with pytest.raises(DatasetError, match="invalid revision"):
        download_beir("tiny", tmp_path, FakeFetcher(), revision=revision)
