from pathlib import Path

import pytest

from rag.domain import Document, Metadata, Page
from rag.errors import IngestionError
from rag.ingestion.base import Parser
from rag.ingestion.enrichers import StaticMetadataEnricher
from rag.ingestion.parse_stage import ParseStage, SourceFile
from rag.storage import JsonArtifactStore


class FakeParser(Parser):
    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on or set()

    def parse(self, path: Path, doc_id: str) -> Document:
        self.calls.append(doc_id)
        if doc_id in self._fail_on:
            raise IngestionError(f"cannot open {path}")
        pages = (Page(page_index=0, text=f"{doc_id} p0"), Page(page_index=1, text="p1"))
        return Document(
            doc_id=doc_id, source_uri=str(path), pages=pages, metadata={"failed_pages": 1}
        )


class TickClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


@pytest.fixture
def store(tmp_path: Path) -> JsonArtifactStore[Document]:
    return JsonArtifactStore(tmp_path / "parsed", Document)


def _stage(parser: Parser, store: JsonArtifactStore[Document]) -> ParseStage:
    catalog: dict[str, Metadata] = {d: {"company": d.split("_")[0]} for d in ("A_1", "B_1", "C_1")}
    return ParseStage(parser, StaticMetadataEnricher(catalog), store, clock=TickClock())


SOURCES = [SourceFile("A_1", Path("a.pdf")), SourceFile("B_1", Path("b.pdf"))]


def test_parses_enriches_and_saves(store: JsonArtifactStore[Document]) -> None:
    report = _stage(FakeParser(), store).run(SOURCES)

    assert report.parsed == ["A_1", "B_1"]
    assert report.pages == 4
    assert report.failed_pages == 2
    assert report.timings_ms == {"A_1": 500.0, "B_1": 500.0}
    saved = store.load("A_1")
    assert saved.metadata["company"] == "A"
    assert saved.pages[0].text == "A_1 p0"


def test_rerun_skips_existing_unless_forced(store: JsonArtifactStore[Document]) -> None:
    parser = FakeParser()
    _stage(parser, store).run(SOURCES)
    second = _stage(parser, store).run(SOURCES)
    forced = _stage(parser, store).run(SOURCES, force=True)

    assert second.skipped == ["A_1", "B_1"]
    assert forced.parsed == ["A_1", "B_1"]
    assert parser.calls == ["A_1", "B_1", "A_1", "B_1"]


def test_failures_are_recorded_and_do_not_stop_the_batch(
    store: JsonArtifactStore[Document],
) -> None:
    sources = [*SOURCES, SourceFile("UNKNOWN_1", Path("u.pdf"))]
    report = _stage(FakeParser(fail_on={"A_1"}), store).run(sources)

    assert report.parsed == ["B_1"]
    assert set(report.failed) == {"A_1", "UNKNOWN_1"}
    assert "no catalog metadata" in report.failed["UNKNOWN_1"]
    assert not store.exists("A_1")
