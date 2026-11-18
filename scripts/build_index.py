"""Embed all chunks and save the exact dense index (HLD 004).

Usage:
    uv run python scripts/build_index.py --config configs/baseline.yaml [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import load_config
from rag.domain import ChunkSet
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.index_stage import IndexStage
from rag.indexing.vector_store import MANIFEST_FILE
from rag.ingestion.factory import build_chunker
from rag.observability import configure_logging
from rag.storage import JsonArtifactStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--force", action="store_true", help="rebuild an existing index")
    args = parser.parse_args(argv)
    configure_logging("WARNING")

    config = load_config(args.config)
    fingerprint = build_chunker(config.chunking).fingerprint
    index_dir = dense_index_dir(config, fingerprint)
    if (index_dir / MANIFEST_FILE).exists() and not args.force:
        print(f"index exists: {index_dir} (use --force to rebuild)")
        return 0

    chunk_sets = JsonArtifactStore(config.paths.chunks_dir(fingerprint), ChunkSet)
    if not chunk_sets.ids():
        print(f"no chunks under {chunk_sets.root}; run scripts/chunk_documents.py first")
        return 1

    embedder = build_embedder(config.embedding)
    store, report = IndexStage(embedder, chunk_sets).build(chunk_sets.ids())
    store.save(index_dir)
    print(
        f"model={embedder.model_id} dim={embedder.dimension} docs={report.documents} "
        f"chunks={report.chunks} truncated={len(report.truncated_chunk_ids)} "
        f"time={report.seconds:.1f}s ({report.chunks_per_second:.0f} chunks/s)\n"
        f"index: {index_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
