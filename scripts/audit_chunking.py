"""Measure whether golden evidence fits inside single chunks (HLD 003).

Usage:
    uv run python scripts/audit_chunking.py --config configs/baseline.yaml [--worst 10]

Writes the full report to runs/chunk_audit/<experiment>.json.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset
from rag.domain import ChunkSet
from rag.evaluation.chunk_audit import ChunkAuditor
from rag.ingestion.factory import build_chunker
from rag.storage import JsonArtifactStore, atomic_write_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    fingerprint = build_chunker(config.chunking, parser=config.parsing.parser).fingerprint
    store = JsonArtifactStore(config.paths.chunks_dir(fingerprint), ChunkSet)
    chunks_by_doc = {doc_id: store.load(doc_id).chunks for doc_id in store.ids()}
    if not chunks_by_doc:
        print(f"no chunks under {store.root}; run scripts/chunk_documents.py first")
        return 1

    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()
    threshold = config.evaluation.parse_audit_min_coverage
    report = ChunkAuditor(chunks_by_doc, threshold).audit(items)

    print(f"chunker {fingerprint}: {report.chunk_count} chunks over {len(chunks_by_doc)} docs")
    print(f"chunk tokens: {report.token_percentiles()}")
    print(
        f"evidence spans: {len(report.rows)}  without chunks: {len(report.missing_doc_rows)}\n"
        f"evidence fitting in ONE chunk (coverage >= {threshold}): "
        f"{report.fits_in_one_chunk_rate:.1%}   mean best coverage: {report.mean_best_coverage:.3f}"
    )

    tags_by_item = {item.item_id: item.tags for item in items}
    by_tag: dict[str, list[float]] = defaultdict(list)
    for row in report.measured:
        for tag in tags_by_item[row.item_id]:
            by_tag[tag].append(row.best_coverage or 0.0)
    print("\nmean best single-chunk coverage per segment:")
    for tag, values in sorted(by_tag.items()):
        print(f"  {tag:<58} n={len(values):>3}  {fmean(values):.3f}")

    print(f"\nworst {args.worst}:")
    for row in sorted(report.measured, key=lambda r: r.best_coverage or 0.0)[: args.worst]:
        print(
            f"  {row.best_coverage:.2f}  {row.item_id}  {row.doc_id} p{row.page_index}"
            f"  ({row.chunks_on_page} chunks on page)"
        )

    out = config.paths.runs_dir / "chunk_audit" / f"{config.experiment_name}.json"
    atomic_write_bytes(out, report.model_dump_json(indent=2).encode("utf-8"))
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
