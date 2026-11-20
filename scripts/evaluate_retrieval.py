"""Evaluate dense retrieval on the FinanceBench golden set (HLD 005).

Usage:
    uv run python scripts/evaluate_retrieval.py --config configs/baseline.yaml [--split dev]

Writes runs/<run_id>/traces.jsonl and runs/<run_id>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.app import load_golden_items, load_retrieval_stack
from rag.config import load_config
from rag.errors import GenerationError
from rag.evaluation.retrieval_report import OVERALL, RetrievalEvaluator
from rag.evaluation.retrieval_runner import (
    RetrievalEvalRunner,
    doc_metadata_from_chunks,
    indexed_units,
)
from rag.evaluation.splits import SplitName
from rag.generation.llm_clients import check_ollama_model
from rag.storage import atomic_write_bytes

SHOWN_METRICS = (
    "recall@{k}",
    "precision@{k}",
    "hit@{k}",
    "ndcg@{k}",
    "mrr",
    "doc_hit@{k}",
    "filter_ok",
    "pool_recall",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--split",
        type=SplitName,
        choices=list(SplitName),
        default=SplitName.DEV,
        help="dev (default, for tuning), test (held out: confirm, never tune) or all",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    planner_config = config.retrieval.rewrite.llm_planner
    if planner_config.enabled:
        try:
            check_ollama_model(planner_config.model_id, planner_config.ollama_url)
        except GenerationError as exc:
            print(f"query planner unavailable: {exc}")
            return 2
    stack = load_retrieval_stack(config)
    store = stack.store
    items = load_golden_items(config, args.split)
    top_k = config.retrieval.top_k
    runner = RetrievalEvalRunner(
        retriever=stack.retriever,
        evaluator=RetrievalEvaluator(config.evaluation.ks),
        indexed=indexed_units(store.chunks),
        top_k=top_k,
        doc_metadata=doc_metadata_from_chunks(store.chunks),
    )
    result = runner.run(items)

    run_id = f"{config.experiment_name}-{args.split}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = config.paths.runs_dir / run_id
    traces = "".join(t.model_dump_json() + "\n" for t in result.traces)
    atomic_write_bytes(out / "traces.jsonl", traces.encode("utf-8"))
    summary = result.model_dump(exclude={"traces"}, mode="json")
    summary |= {
        "run_id": run_id,
        "split": str(args.split),
        "config": config.model_dump(mode="json"),
    }
    if (planner := stack.planner) is not None:
        summary["planner"] = {
            "calls": planner.calls,
            "cache_hits": planner.cache_hits,
            "failures": planner.failures,
        }
    atomic_write_bytes(out / "metrics.json", _json(summary))

    names = [m.format(k=top_k) for m in SHOWN_METRICS]
    print(f"run {run_id}: {len(result.traces)} questions")
    print(f"status @top{top_k}: {dict(result.status_counts)}")
    filtered = sum(bool(t.applied_filters) for t in result.traces)
    relaxed = sum(bool(t.relaxed) for t in result.traces)
    print(f"filters applied: {filtered}/{len(result.traces)} questions, relaxed: {relaxed}")
    if planner is not None:
        print(
            f"query planner: {planner.calls} LLM calls, {planner.cache_hits} cache hits, "
            f"{planner.failures} failures (fell back to rules)"
        )
    print(f"retrieval latency ms: { {k: round(v, 1) for k, v in result.latency_ms.items()} }")
    for stage, summary_ms in result.stage_latency_ms.items():
        print(f"  {stage:<12} ms: { {k: round(v, 1) for k, v in summary_ms.items()} }")
    print()
    print(f"{'segment':<58} {'n':>4}  " + "  ".join(f"{n:>12}" for n in names))
    for segment, summary_row in result.report.segments.items():
        marker = "*" if segment == OVERALL else " "
        values = "  ".join(f"{summary_row.means[n]:>12.3f}" for n in names)
        print(f"{marker}{segment:<57} {summary_row.count:>4}  {values}")
    print(f"\ntraces: {out / 'traces.jsonl'}")
    return 0


def _json(data: object) -> bytes:
    return json.dumps(data, indent=2, default=str).encode("utf-8")


if __name__ == "__main__":
    sys.exit(main())
