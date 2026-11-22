"""Download BEIR datasets into data/raw/beir (HLD 025).

Usage:
    uv run python scripts/download_beir.py [scifact nfcorpus fiqa]

Safe to re-run: datasets already extracted are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.datasets.beir import download_beir
from rag.datasets.http import UrllibFetcher

DEFAULT_ROOT = Path("data/raw/beir")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--published", type=Path, default=Path("evals/beir-published.json"))
    args = parser.parse_args(argv)
    spec = json.loads(args.published.read_text(encoding="utf-8"))
    fetcher = UrllibFetcher(timeout_s=300)
    for name in args.names:
        revision = spec["datasets"][name]["mteb_revision"]
        path = download_beir(name, args.root, fetcher, revision, split=spec["split"])
        print(f"{name}: {path} (mteb/{name}@{revision[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
