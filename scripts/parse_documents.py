"""Parse downloaded FinanceBench PDFs into data/processed/parsed/<doc_id>.json.

Usage:
    uv run python scripts/parse_documents.py --config configs/baseline.yaml [--force]

Parses every downloaded PDF that is in the FinanceBench catalog. Safe to re-run:
already-parsed documents are skipped unless --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset, document_metadata
from rag.domain import Document
from rag.ingestion.enrichers import StaticMetadataEnricher
from rag.ingestion.factory import build_parser
from rag.ingestion.parse_stage import ParseStage, SourceFile
from rag.observability import configure_logging
from rag.storage import JsonArtifactStore

logger = logging.getLogger("parse_documents")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--force", action="store_true", help="re-parse existing artifacts")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    dataset = FinanceBenchDataset(config.paths.raw_dir / "financebench")
    catalog = {doc.doc_name: document_metadata(doc) for doc in dataset.documents()}
    sources = [
        SourceFile(doc_id=name, path=dataset.pdf_path(name))
        for name in sorted(catalog)
        if dataset.pdf_path(name).exists()
    ]
    stage = ParseStage(
        parser=build_parser(config.parsing),
        enricher=StaticMetadataEnricher(catalog),
        store=JsonArtifactStore(config.paths.parsed_dir_for(config.parsing.parser), Document),
    )
    report = stage.run(sources, force=args.force)

    total_s = sum(report.timings_ms.values()) / 1000
    print(
        f"\nparsed={len(report.parsed)} skipped={len(report.skipped)} "
        f"failed={len(report.failed)} pages={report.pages} "
        f"failed_pages={report.failed_pages} time={total_s:.1f}s"
    )
    if report.parsed:
        print(f"mean {total_s * 1000 / max(report.pages, 1):.1f} ms/page")
    for doc_id, error in report.failed.items():
        print(f"  FAILED {doc_id}: {error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
