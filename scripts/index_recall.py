"""Index recall: how often approximate search (HNSW) finds the exact top-k (backlog 4.2; HLD 028).

Usage:
    uv sync --group ann        # faiss-cpu, the reference HNSW
    uv run python scripts/index_recall.py                       # FinanceBench dev + FiQA
    uv run python scripts/index_recall.py --corpora financebench --m 16

Index recall@k = |ANN top-k ∩ exact top-k| / k, averaged over real queries. The exact
store is the ground truth (ADR 0003), so no relevance labels are involved: this measures
the index, not retrieval quality. For each corpus and ``M`` it builds our from-scratch
graph (``rag.indexing.hnsw``) and faiss ``IndexHNSWFlat`` with the same ``M`` and
``ef_construction``, then sweeps ``ef_search``. It reports recall, vectors compared per
query and single-query latency. On FinanceBench it also measures filtered search with the
company / year filters the planner-rules config extracts from each question, using three
strategies: post-filtering, filtering inside the graph walk, and the exact fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
import numpy.typing as npt

from rag.app import load_golden_items
from rag.config import AppConfig, load_config
from rag.datasets.beir import BeirDataset
from rag.datasets.financebench import COMPANY_ALIASES
from rag.domain import Filters
from rag.evaluation.splits import SplitName
from rag.indexing.factory import build_embedder, dense_index_dir, model_slug
from rag.indexing.hnsw import HNSWGraph
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.observability import configure_logging
from rag.retrieval.filters import CatalogFilterExtractor
from rag.storage import atomic_write_bytes

Floats = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]


@dataclass
class Point:
    corpus: str
    index: str
    m: int
    ef_search: int
    recall: float
    compared_per_query: float | None
    p50_ms: float
    build_s: float


def exact_top(vectors: Floats, queries: Floats, k: int) -> list[set[int]]:
    sims = queries @ vectors.T
    return [{int(i) for i in np.argsort(-row, kind="stable")[:k]} for row in sims]


def recall(found: list[list[int]], truth: list[set[int]], k: int) -> float:
    return fmean(
        len(set(f[:k]) & t) / min(k, len(t)) if t else 1.0
        for f, t in zip(found, truth, strict=True)
    )


def timed_queries(
    search: Callable[[Floats], list[int]], queries: Floats
) -> tuple[list[list[int]], float]:
    found, times = [], []
    for q in queries:
        start = time.perf_counter()
        found.append(search(q))
        times.append((time.perf_counter() - start) * 1000)
    return found, median(times)


def exact_search(vectors: Floats, k: int) -> Callable[[Floats], list[int]]:
    return lambda q: [int(i) for i in np.argsort(-(vectors @ q))[:k]]


def ours_search(graph: HNSWGraph, k: int, ef: int) -> Callable[[Floats], list[int]]:
    return lambda q: [n for _, n in graph.search(q, k, ef)]


def faiss_search(index: Any, k: int) -> Callable[[Floats], list[int]]:
    return lambda q: [int(i) for i in index.search(q[None, :], k)[1][0] if i >= 0]


def load_corpus(name: str, config: AppConfig) -> tuple[ExactVectorStore, Floats, list[Filters]]:
    """Vectors of the indexed corpus, embedded queries and (FinanceBench) their filters."""
    embedder = build_embedder(config.embedding)
    if name == "financebench":
        fingerprint = build_chunker(config.chunking, parser=config.parsing.parser).fingerprint
        store = ExactVectorStore.load(
            dense_index_dir(config, fingerprint), expected_model_id=embedder.model_id
        )
        items = load_golden_items(config, SplitName.DEV)
        f = config.retrieval.filters
        extractor = CatalogFilterExtractor(
            COMPANY_ALIASES,
            year_window=f.year_window,
            short_fiscal_years=f.short_fiscal_years,
            forward_looking_prior_year=f.forward_looking_prior_year,
        )
        texts = [item.question for item in items]
        filters = [extractor.extract(t) for t in texts]
    else:
        index_dir = config.paths.data_dir / "indexes" / "beir"
        store = ExactVectorStore.load(
            index_dir / f"{name}__{model_slug(embedder.model_id)}",
            expected_model_id=embedder.model_id,
        )
        texts = list(BeirDataset(Path("data/raw/beir") / name).queries().values())
        filters = []
    queries = np.stack([embedder.embed_query(t) for t in texts]).astype(np.float32)
    return store, queries, filters


def sweep(
    corpus: str, vectors: Floats, queries: Floats, args: argparse.Namespace
) -> tuple[list[Point], dict[int, HNSWGraph]]:
    import faiss

    faiss.omp_set_num_threads(1)  # single-query latency, like one user request
    k = args.k
    truth = exact_top(vectors, queries, k)
    points = []
    _, exact_ms = timed_queries(exact_search(vectors, k), queries)
    points.append(Point(corpus, "exact", 0, 0, 1.0, float(len(vectors)), exact_ms, 0.0))
    graphs = {}
    for m in args.m:
        start = time.perf_counter()
        graph = HNSWGraph(vectors, m=m, ef_construction=args.ef_construction, seed=0)
        ours_build = time.perf_counter() - start
        graphs[m] = graph
        start = time.perf_counter()
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
        index.add(vectors)
        faiss_build = time.perf_counter() - start
        print(f"  {corpus} M={m}: build ours {ours_build:.1f} s, faiss {faiss_build:.1f} s")
        for ef in args.ef:
            graph.distance_computations = 0
            found, ms = timed_queries(ours_search(graph, k, ef), queries)
            compared = graph.distance_computations / len(queries)
            points.append(
                Point(corpus, "ours", m, ef, recall(found, truth, k), compared, ms, ours_build)
            )
            index.hnsw.efSearch = max(ef, k)
            found, ms = timed_queries(faiss_search(index, k), queries)
            points.append(
                Point(corpus, "faiss", m, ef, recall(found, truth, k), None, ms, faiss_build)
            )
    return points, graphs


def filtered(
    store: ExactVectorStore,
    graph: HNSWGraph,
    queries: Floats,
    filters: list[Filters],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Filtered index recall: post-filter vs in-walk filter vs exact fallback."""
    vectors = store.vectors
    k = args.k
    rows: list[dict[str, object]] = []
    cases = [(q, store.filter_mask(f)) for q, f in zip(queries, filters, strict=True) if f]
    sizes = [int(mask.sum()) for _, mask in cases]
    truths = []
    for q, mask in cases:
        sims = np.where(mask, vectors @ q, -np.inf)
        truths.append({int(i) for i in np.argsort(-sims, kind="stable")[:k] if mask[i]})
    for ef in args.ef:
        post, walk, compared = [], [], 0
        for q, mask in cases:
            # post-filter: search unfiltered, then drop disallowed hits
            post.append([n for _, n in graph.search(q, ef, ef=ef) if mask[n]][:k])
            graph.distance_computations = 0
            walk.append([n for _, n in graph.search(q, k, ef=ef, allowed=mask)])
            compared += graph.distance_computations
        rows.append(
            {
                "ef_search": ef,
                "post_filter_recall": recall(post, truths, k),
                "in_walk_recall": recall(walk, truths, k),
                "in_walk_compared_per_query": compared / len(cases),
            }
        )
    fallback = sum(s <= args.exact_below for s in sizes)
    print(
        f"  filtered: {len(cases)}/{len(queries)} questions have filters; allowed chunks "
        f"median {median(sizes):.0f} (min {min(sizes)}, max {max(sizes)}, of {len(store)}); "
        f"{fallback} fall back to exact search (exact_below={args.exact_below})"
    )
    for row in rows:
        print(
            f"    ef {row['ef_search']:>4}: post-filter {row['post_filter_recall']:.3f}  "
            f"in-walk {row['in_walk_recall']:.3f} "
            f"({row['in_walk_compared_per_query']:.0f} compared/query)"
        )
    rows.append(
        {
            "questions_filtered": len(cases),
            "allowed_median": median(sizes),
            "allowed_min": min(sizes),
            "allowed_max": max(sizes),
            "exact_fallbacks": fallback,
            "exact_below": args.exact_below,
        }
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/planner-rules.yaml"))
    parser.add_argument("--corpora", nargs="+", default=["financebench", "fiqa"])
    parser.add_argument("--m", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 16, 32, 64, 128, 256])
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exact-below", type=int, default=2000)
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    config = load_config(args.config)

    summary: dict[str, object] = {"args": {k: v for k, v in vars(args).items() if k != "config"}}
    summary["config"] = str(args.config)
    all_points: list[Point] = []
    for corpus in args.corpora:
        store, queries, filters = load_corpus(corpus, config)
        print(f"{corpus}: {len(store)} vectors, {len(queries)} queries, k={args.k}")
        points, graphs = sweep(corpus, store.vectors, queries, args)
        all_points += points
        print(f"  {'index':<6}{'M':>4}{'ef':>6}{'recall@k':>10}{'compared':>10}{'p50 ms':>9}")
        for p in points:
            compared = f"{p.compared_per_query:.0f}" if p.compared_per_query else "-"
            row = f"  {p.index:<6}{p.m:>4}{p.ef_search:>6}{p.recall:>10.3f}"
            print(f"{row}{compared:>10}{p.p50_ms:>9.2f}")
        if filters:
            middle = args.m[len(args.m) // 2]
            summary[f"{corpus}_filtered_m{middle}"] = filtered(
                store, graphs[middle], queries, filters, args
            )
    summary["points"] = [asdict(p) for p in all_points]
    out = (
        config.paths.runs_dir / f"index-recall-{datetime.now(UTC):%Y%m%dT%H%M%SZ}" / "summary.json"
    )
    atomic_write_bytes(out, json.dumps(summary, indent=2).encode())
    print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
