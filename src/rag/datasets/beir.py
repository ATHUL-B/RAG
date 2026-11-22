"""BEIR datasets (SciFact, NFCorpus, FiQA-2018) for cross-checking retrieval code
against published numbers (backlog 4.4; HLD 025).

A BEIR dataset is a corpus (``corpus.jsonl``: _id, title, text), queries (``queries.jsonl``)
and graded relevance judgments (``qrels/<split>.tsv``: query-id, corpus-id, score). Each
document becomes one :class:`Chunk` whose text is ``title + " " + text``, the same text
MTEB embeds, so our embedder, exact index and metric functions run unchanged and can be
compared with the model card's MTEB scores.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Final

from rag.datasets.http import Fetcher
from rag.domain import Chunk
from rag.errors import DatasetError

MTEB_URL: Final = "https://huggingface.co/datasets/mteb/{name}/resolve/{revision}/{path}"
"""MTEB's copies of the BEIR datasets: the exact data the published scores were computed on.
(The original BEIR mirror at public.ukp.informatik.tu-darmstadt.de timed out on
2026-11-22.)"""

Qrels = dict[str, dict[str, int]]


class BeirDataset:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def corpus(self) -> list[Chunk]:
        chunks = []
        for row in _jsonl(self._dir / "corpus.jsonl"):
            # FiQA has 38 empty documents (one judged relevant). MTEB embeds them as "".
            # A single space tokenizes to the same [CLS][SEP], so the embedding is
            # identical, and a Chunk never has empty text.
            text = f"{row.get('title', '')} {row.get('text', '')}".strip() or " "
            doc_id = str(row["_id"])
            chunks.append(
                Chunk(
                    chunk_id=doc_id,
                    doc_id=doc_id,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    page_start=0,
                    page_end=0,
                )
            )
        return chunks

    def qrels(self, split: str = "test") -> Qrels:
        """query-id → {corpus-id: grade}. Grades <= 0 are kept (they are judged, not relevant)."""
        path = self._dir / "qrels" / f"{split}.tsv"
        if not path.exists():
            raise DatasetError(f"{path} not found; run scripts/download_beir.py first")
        qrels: Qrels = defaultdict(dict)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            parts = line.split("\t")
            if line_no == 0 and parts[0] == "query-id":
                continue  # header
            if len(parts) != 3:
                raise DatasetError(f"{path}:{line_no + 1}: expected 3 tab-separated fields")
            qrels[parts[0]][parts[1]] = int(parts[2])
        return dict(qrels)

    def queries(self, split: str = "test") -> dict[str, str]:
        """query-id → text, only for queries judged in ``split`` (as BEIR / MTEB evaluate)."""
        judged = self.qrels(split)
        return {
            str(row["_id"]): str(row["text"])
            for row in _jsonl(self._dir / "queries.jsonl")
            if str(row["_id"]) in judged
        }


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise DatasetError(f"{path} not found; run scripts/download_beir.py first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_beir(
    name: str, root: Path, fetcher: Fetcher, revision: str, split: str = "test"
) -> Path:
    """Fetch ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/<split>.tsv`` of ``mteb/<name>``
    at a pinned ``revision`` into ``root/<name>``. Files already present are skipped."""
    if not revision or "/" in revision or ".." in revision:
        raise DatasetError(f"invalid revision for {name}: {revision!r}")
    target = root / name
    for path in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"):
        destination = target / path
        if destination.exists():
            continue
        body = fetcher.fetch(MTEB_URL.format(name=name, revision=revision, path=path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    return target
