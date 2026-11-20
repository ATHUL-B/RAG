"""Create the committed dev/test split of the FinanceBench golden set (backlog: dev/test).

Usage:
    uv run python scripts/make_split.py [--config configs/baseline.yaml] [--force]

Writes evals/financebench-split.json (question IDs only). The split is meant to be made
**once**: regenerating it after looking at test results would leak test into tuning, so an
existing file is never overwritten without --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset
from rag.evaluation.splits import make_split, save_split

TEST_FRACTION = 1 / 3
SEED = 0  # fixed before looking at any split; never shop for a "nicer" seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    path = config.evaluation.split_file
    if path.exists() and not args.force:
        print(f"{path} exists; the split is frozen (use --force only if you know why)")
        return 1
    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()
    split = make_split(items, test_fraction=TEST_FRACTION, seed=SEED).model_copy(
        update={
            "notes": (
                "Tune on dev only. Evaluate test rarely, to confirm a change generalises; "
                "never pick settings by their test score."
            )
        }
    )
    save_split(split, path)
    companies = {i.metadata["company"] for i in items if i.item_id in set(split.test)}
    print(f"dev={len(split.dev)} test={len(split.test)} ({len(companies)} test companies)")
    for name, strata in split.strata.items():
        print(f"  {name}: {strata}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
