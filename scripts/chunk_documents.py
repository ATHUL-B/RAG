"""Chunk parsed documents into data/processed/chunks/<fingerprint>/<doc_id>.json.

Usage:
    uv run python scripts/chunk_documents.py --config configs/baseline.yaml [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag.config import load_config
from rag.domain import ChunkSet, Document
from rag.ingestion.chunk_stage import ChunkStage
from rag.ingestion.factory import build_chunker
from rag.observability import configure_logging
from rag.storage import JsonArtifactStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--force", action="store_true", help="re-chunk existing artifacts")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    chunker = build_chunker(config.chunking, parser=config.parsing.parser)
    documents = JsonArtifactStore(config.paths.parsed_dir_for(config.parsing.parser), Document)
    chunk_sets = JsonArtifactStore(config.paths.chunks_dir(chunker.fingerprint), ChunkSet)

    started = time.perf_counter()
    report = ChunkStage(chunker, documents, chunk_sets).run(documents.ids(), force=args.force)
    elapsed = time.perf_counter() - started

    print(
        f"\nchunker={chunker.fingerprint} chunked={len(report.chunked)} "
        f"skipped={len(report.skipped)} failed={len(report.failed)} "
        f"chunks={report.chunks} time={elapsed:.1f}s\nartifacts: {chunk_sets.root}"
    )
    for doc_id, error in report.failed.items():
        print(f"  FAILED {doc_id}: {error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
