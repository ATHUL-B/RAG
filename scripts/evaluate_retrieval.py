"""Evaluate dense retrieval on the FinanceBench golden set (HLD 005).

Usage:
    uv run python scripts/evaluate_retrieval.py --config configs/baseline.yaml

Writes runs/<run_id>/traces.jsonl and runs/<run_id>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset
from rag.evaluation.retrieval_report import OVERALL, RetrievalEvaluator
from rag.evaluation.retrieval_runner import RetrievalEvalRunner, indexed_units
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.retrieval.dense import DenseRetriever
from rag.storage import atomic_write_bytes

SHOWN_METRICS = ("recall@{k}", "precision@{k}", "hit@{k}", "ndcg@{k}", "mrr", "doc_hit@{k}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    embedder = build_embedder(config.embedding)
    index_dir = dense_index_dir(config, build_chunker(config.chunking).fingerprint)
    store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()

    embedder.embed_query("warm-up")  # load the model outside the timed loop
    top_k = config.retrieval.top_k
    runner = RetrievalEvalRunner(
        retriever=DenseRetriever(embedder, store),
        evaluator=RetrievalEvaluator(config.evaluation.ks),
        indexed=indexed_units(store.chunks),
        top_k=top_k,
    )
    result = runner.run(items)

    run_id = f"{config.experiment_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = config.paths.runs_dir / run_id
    traces = "".join(t.model_dump_json() + "\n" for t in result.traces)
    atomic_write_bytes(out / "traces.jsonl", traces.encode("utf-8"))
    summary = result.model_dump(exclude={"traces"}, mode="json")
    summary |= {"run_id": run_id, "config": config.model_dump(mode="json")}
    atomic_write_bytes(out / "metrics.json", _json(summary))

    names = [m.format(k=top_k) for m in SHOWN_METRICS]
    print(f"run {run_id}: {len(result.traces)} questions, index {index_dir.name}")
    print(f"status @top{top_k}: {dict(result.status_counts)}")
    print(f"retrieval latency ms: { {k: round(v, 1) for k, v in result.latency_ms.items()} }\n")
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
