"""Cross-check our dense retrieval and metrics on BEIR against published numbers (HLD 025).

Usage:
    uv run python scripts/evaluate_beir.py [scifact nfcorpus fiqa] [--config <config>]

Uses the config's embedding model and query prefix, our ExactVectorStore and our metric
functions (nDCG@10, recall@100), and compares them with evals/beir-published.json (the
model card's MTEB results for the same model revision). Corpus embeddings are cached
under data/indexes/beir/. Exit 1 if any dataset is outside the tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from rag.config import load_config
from rag.datasets.beir import BeirDataset
from rag.domain import Query
from rag.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
from rag.indexing.factory import build_embedder, model_slug
from rag.indexing.vector_store import ExactVectorStore
from rag.retrieval.dense import DenseRetriever
from rag.storage import atomic_write_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--config", type=Path, default=Path("configs/planner-rules.yaml"))
    parser.add_argument("--published", type=Path, default=Path("evals/beir-published.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/beir"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    published = json.loads(args.published.read_text(encoding="utf-8"))
    tolerance = float(published["tolerance"])
    embedder = build_embedder(config.embedding)
    summary: dict[str, dict[str, float]] = {}
    ok = True
    print(f"{'dataset':<10}{'docs':>8}{'queries':>9}  {'nDCG@10 ours / published':>28}", end="")
    print(f"  {'R@100 ours / published':>28}")
    for name in args.names:
        dataset = BeirDataset(args.root / name)
        index_dir = (
            config.paths.data_dir / "indexes" / "beir" / f"{name}__{model_slug(embedder.model_id)}"
        )
        if index_dir.exists():
            store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
        else:
            chunks = dataset.corpus()
            start = time.perf_counter()
            vectors = embedder.embed_documents([c.indexed_text for c in chunks])
            store = ExactVectorStore(embedder.model_id, embedder.dimension)
            store.upsert(chunks, vectors)
            store.save(index_dir)
            print(f"  embedded {len(chunks)} {name} docs in {time.perf_counter() - start:.0f} s")
        retriever = DenseRetriever(embedder, store)
        qrels = dataset.qrels(published["split"])
        ndcg, recall, skipped = [], [], 0
        for qid, text in dataset.queries(published["split"]).items():
            grades = qrels[qid]
            if not any(g > 0 for g in grades.values()):
                skipped += 1  # no relevant document: nDCG and recall undefined
                continue
            ranked = [
                c.chunk.chunk_id for c in retriever.retrieve(Query(text=text), 100).candidates
            ]
            ndcg.append(ndcg_at_k(ranked, grades, 10))
            recall.append(recall_at_k(ranked, grades, 100))
        ours = {"ndcg@10": fmean(ndcg), "recall@100": fmean(recall)}
        ref = published["datasets"][name]
        within = all(abs(ours[m] - ref[m]) <= tolerance for m in ours)
        ok &= within
        summary[name] = {
            **ours,
            "queries": len(ndcg),
            "skipped": skipped,
            "docs": len(store),
            **{f"published_{m}": ref[m] for m in ours},
        }
        cells = "  ".join(_cell(ours[m], ref[m]) for m in ("ndcg@10", "recall@100"))
        verdict = "OK" if within else f"OUTSIDE ±{tolerance}"
        print(f"{name:<10}{len(store):>8}{len(ndcg):>9}  {cells}  {verdict}")
    out = config.paths.runs_dir / f"beir-{datetime.now(UTC):%Y%m%dT%H%M%SZ}" / "summary.json"
    atomic_write_bytes(
        out, json.dumps({"model_id": embedder.model_id, "results": summary}, indent=2).encode()
    )
    print(f"\nsummary: {out}")
    return 0 if ok else 1


def _cell(ours: float, published: float) -> str:
    return f"{ours:>8.4f} / {published:.4f} ({ours - published:+.4f})"


if __name__ == "__main__":
    sys.exit(main())
