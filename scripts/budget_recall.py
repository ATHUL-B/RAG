"""Recall at an equal prompt-token budget, across retrieval runs (HLD 022).

Usage:
    uv run python scripts/budget_recall.py runs/<run-a> runs/<run-b> ... \
        [--budgets 1000 2000 4000] [--baseline runs/<run>]

recall@k compares runs at the same *number* of chunks, which is unfair when chunk sizes
differ. This walks each question's ranking (from traces.jsonl), adds chunks while their
prompt tokens (the compacted context header + text, as the generator sees them) fit the
budget, and reports mean recall per budget. With ``--baseline``, each run is also compared
with it (paired bootstrap 95% CI, as in compare_runs.py). Runs must be deep enough:
evaluate with ``ks`` up to 20.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

from rag.config import AppConfig
from rag.domain import ChunkSet
from rag.evaluation.compare import paired_bootstrap_ci
from rag.evaluation.retrieval_metrics import recall_at_token_budget
from rag.generation.context import compact
from rag.ingestion.factory import build_chunker
from rag.ingestion.tokenization import TiktokenCounter
from rag.storage import JsonArtifactStore


def run_scores(run: Path, budgets: list[int]) -> tuple[dict[int, dict[str, float]], float]:
    """Per budget: question → recall. Also the mean prompt tokens of the top 5."""
    config = AppConfig.model_validate(json.loads((run / "metrics.json").read_text())["config"])
    fingerprint = build_chunker(config.chunking, parser=config.parsing.parser).fingerprint
    store = JsonArtifactStore(config.paths.chunks_dir(fingerprint), ChunkSet)
    counter = TiktokenCounter(config.chunking.tokenizer)
    tokens: dict[str, int] = {}
    for key in store.ids():
        for chunk in store.load(key).chunks:
            tokens[chunk.chunk_id] = counter.count(compact(chunk.indexed_text))
    scores: dict[int, dict[str, float]] = {b: {} for b in budgets}
    top5: list[int] = []
    for line in (run / "traces.jsonl").read_text().splitlines():
        trace = json.loads(line)
        qrels = dict.fromkeys(trace["evidence_units"], 1)
        if not qrels:
            continue
        ranked = [r["relevance_id"] for r in trace["retrieved"]]
        counts = [tokens[r["chunk_id"]] for r in trace["retrieved"]]
        top5.append(sum(counts[:5]))
        for budget in budgets:
            scores[budget][trace["item_id"]] = recall_at_token_budget(ranked, counts, qrels, budget)
    return scores, fmean(top5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000])
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args(argv)

    base = run_scores(args.baseline, args.budgets)[0] if args.baseline else None
    header = "".join(f"{f'@{b} tok':>12}" for b in args.budgets)
    print(f"{'run':<48}{'top-5 tokens':>13}{header}")
    for run in args.runs:
        scores, top5 = run_scores(run, args.budgets)
        cells = "".join(f"{fmean(scores[b].values()):>12.3f}" for b in args.budgets)
        print(f"{run.name:<48}{top5:>13.0f}{cells}")
        if base is not None and run != args.baseline:
            for budget in args.budgets:
                ids = sorted(set(scores[budget]) & set(base[budget]))
                diffs = [scores[budget][i] - base[budget][i] for i in ids]
                low, high = paired_bootstrap_ci(diffs)
                star = " *" if low > 0 or high < 0 else ""
                print(
                    f"    @{budget}: {fmean(diffs):+.3f} [{low:+.3f}, {high:+.3f}]{star}"
                    f"  vs {args.baseline.name}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
