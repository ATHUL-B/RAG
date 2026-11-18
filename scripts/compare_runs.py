"""Compare two evaluation runs per question, with paired-bootstrap confidence intervals.

Usage:
    uv run python scripts/compare_runs.py runs/filters-<ts> runs/hybrid-<ts> \
        [--metrics recall@5 ndcg@5 mrr]

Positive differences mean the SECOND run is better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.evaluation.compare import compare_runs

DEFAULT_METRICS = ("recall@5", "hit@5", "ndcg@5", "mrr", "recall@10")


def load_scores(run_dir: Path, metric: str) -> dict[str, float]:
    with (run_dir / "traces.jsonl").open(encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    return {t["item_id"]: float(t["metrics"][metric]) for t in traces}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    args = parser.parse_args(argv)

    print(f"A = {args.run_a.name}\nB = {args.run_b.name}\n")
    print(f"{'metric':<12} {'A':>7} {'B':>7} {'B-A':>8}  {'95% CI':>18}  better/worse/same")
    for metric in args.metrics:
        result = compare_runs(
            load_scores(args.run_a, metric), load_scores(args.run_b, metric), metric
        )
        flag = "  *" if result.significant else ""
        print(
            f"{metric:<12} {result.mean_a:>7.3f} {result.mean_b:>7.3f} {result.mean_diff:>+8.3f}"
            f"  [{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
            f"  {result.better}/{result.worse}/{result.same}{flag}"
        )
    print("\n* = 95% interval excludes 0 (difference unlikely to be noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
