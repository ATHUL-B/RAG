"""Download FinanceBench into data/raw/financebench.

Usage:
    uv run python scripts/download_financebench.py --scope questions   # 84 PDFs (default)
    uv run python scripts/download_financebench.py --scope all         # all ~368 PDFs, ~700 MB
    uv run python scripts/download_financebench.py --scope metadata    # JSONL files only

Safe to re-run: files already present are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from enum import StrEnum
from pathlib import Path

from rag.datasets.financebench import FinanceBenchDataset, FinanceBenchDownloader
from rag.datasets.http import UrllibFetcher
from rag.observability import configure_logging

DEFAULT_ROOT = Path("data/raw/financebench")
logger = logging.getLogger("download_financebench")


class Scope(StrEnum):
    METADATA = "metadata"
    QUESTIONS = "questions"
    ALL = "all"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scope", type=Scope, choices=list(Scope), default=Scope.QUESTIONS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    configure_logging()

    downloader = FinanceBenchDownloader(UrllibFetcher(), args.root)
    report = downloader.download_metadata()
    if report.failed:
        logger.error("metadata download failed", extra={"failed": report.failed})
        return 1

    dataset = FinanceBenchDataset(args.root)
    if args.scope is Scope.QUESTIONS:
        report = downloader.download_pdfs(dataset.referenced_doc_names())
    elif args.scope is Scope.ALL:
        report = downloader.download_pdfs(d.doc_name for d in dataset.documents())
    else:
        return 0

    logger.info(
        "pdf download finished",
        extra={
            "downloaded": len(report.downloaded),
            "skipped": len(report.skipped),
            "failed": report.failed,
        },
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
