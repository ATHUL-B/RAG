"""Measure how much golden evidence survived PDF parsing (debugging-ladder rung 1).

Usage:
    uv run python scripts/audit_parsing.py --config configs/baseline.yaml [--worst 10]

Writes the full per-evidence report to runs/parse_audit/<experiment>.json.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset
from rag.domain import Document
from rag.evaluation.parse_audit import ParseAuditor
from rag.storage import JsonArtifactStore, atomic_write_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--worst", type=int, default=10, help="show the N lowest-coverage items")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()
    store = JsonArtifactStore(config.paths.parsed_dir_for(config.parsing.parser), Document)
    report = ParseAuditor(store, config.evaluation.parse_audit_min_coverage).audit(items)

    print(f"evidence spans: {len(report.rows)}  status: {report.status_counts}")
    print(
        f"pass rate (coverage >= {report.min_coverage}): {report.pass_rate:.1%}   "
        f"mean coverage: {report.mean_coverage:.3f}"
    )

    tags_by_item = {item.item_id: item.tags for item in items}
    by_tag: dict[str, list[float]] = defaultdict(list)
    for row in report.measured:
        for tag in tags_by_item[row.item_id]:
            by_tag[tag].append(row.coverage or 0.0)
    print("\nmean coverage per segment:")
    for tag, values in sorted(by_tag.items()):
        print(f"  {tag:<45} n={len(values):>3}  {fmean(values):.3f}")

    print(f"\nworst {args.worst}:")
    for row in report.worst(args.worst):
        hint = f"  (page {row.best_neighbour[0]} matches better)" if row.best_neighbour else ""
        print(f"  {row.coverage:.2f}  {row.item_id}  {row.doc_id} p{row.page_index}{hint}")

    out = config.paths.runs_dir / "parse_audit" / f"{config.experiment_name}.json"
    atomic_write_bytes(out, report.model_dump_json(indent=2).encode("utf-8"))
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
