import itertools

import numpy as np
import numpy.typing as npt
import pytest

from rag.app import dense_search
from rag.config import DenseIndexConfig, DenseIndexKind
from rag.domain import Chunk, Filters
from rag.indexing.hnsw import HNSWGraph, HNSWVectorStore
from rag.indexing.vector_store import ExactVectorStore

Floats = npt.NDArray[np.float32]


def _normalize(x: npt.ArrayLike) -> Floats:
    a = np.asarray(x, dtype=np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _clustered(n: int, dim: int = 32, clusters: int = 20, seed: int = 0) -> Floats:
    """Clustered vectors like real embeddings (uniform random ones are unrealistically hard)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim))
    return _normalize(centres[rng.integers(clusters, size=n)] + 0.3 * rng.normal(size=(n, dim)))


def _exact_top(
    vectors: Floats, q: Floats, k: int, allowed: npt.NDArray[np.bool_] | None = None
) -> set[int]:
    sims = vectors @ q
    if allowed is not None:
        sims = np.where(allowed, sims, -np.inf)
    return {int(i) for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])}


VECTORS = _clustered(1500)
QUERIES = _clustered(60, seed=7)


@pytest.fixture(scope="module")
def graph() -> HNSWGraph:
    return HNSWGraph(VECTORS, m=8, ef_construction=64, seed=0)


def _recall(graph: HNSWGraph, ef: int, k: int = 10) -> float:
    hits = 0
    for q in QUERIES:
        got = {n for _, n in graph.search(q, k, ef)}
        hits += len(got & _exact_top(VECTORS, q, k))
    return hits / (k * len(QUERIES))


def test_recall_is_high_and_rises_with_ef(graph: HNSWGraph) -> None:
    graph.distance_computations = 0
    low = _recall(graph, ef=10)
    low_cost = graph.distance_computations
    graph.distance_computations = 0
    high = _recall(graph, ef=100)
    assert high >= 0.98 and high >= low
    assert graph.distance_computations > low_cost  # a wider beam compares more vectors
    # ...but still far fewer than exact search (1500 per query)
    assert graph.distance_computations / len(QUERIES) < 0.7 * len(VECTORS)


def test_results_are_sorted_true_similarities(graph: HNSWGraph) -> None:
    hits = graph.search(QUERIES[0], 5, ef=32)
    sims = [s for s, _ in hits]
    assert sims == sorted(sims, reverse=True) and len(hits) == 5
    for sim, node in hits:
        assert sim == pytest.approx(float(VECTORS[node] @ QUERIES[0]), abs=1e-5)


def test_layers_shrink_and_degrees_are_capped(graph: HNSWGraph) -> None:
    sizes = graph.layer_sizes()
    assert sizes[0] == len(VECTORS)
    assert all(a > b for a, b in itertools.pairwise(sizes))  # ~1/m per layer
    for node in range(len(VECTORS)):
        assert 1 <= len(graph.neighbours(node)) <= 16  # 2m on layer 0
    for layer in range(1, graph.max_level + 1):
        members = set(graph._links[layer])
        assert members <= set(graph._links[layer - 1])  # every upper node is also below
        assert all(set(graph.neighbours(n, layer)) <= members for n in members)
        assert all(len(graph.neighbours(n, layer)) <= 8 for n in members)


def test_same_seed_same_graph() -> None:
    a = HNSWGraph(VECTORS[:300], m=4, seed=3)
    b = HNSWGraph(VECTORS[:300], m=4, seed=3)
    assert a._links == b._links


def test_allowed_mask_returns_only_allowed_nodes(graph: HNSWGraph) -> None:
    allowed = np.zeros(len(VECTORS), dtype=np.bool_)
    allowed[::3] = True
    for q in QUERIES[:10]:
        got = [n for _, n in graph.search(q, 5, ef=64, allowed=allowed)]
        assert got and all(allowed[n] for n in got)
        assert len(set(got) & _exact_top(VECTORS, q, 5, allowed)) >= 4


def test_neighbour_heuristic_skips_redundant_directions() -> None:
    # base = e0; a and b point the same way, c points elsewhere but is less similar.
    vectors = _normalize(
        np.array([[1, 0, 0], [0.9, 0.44, 0], [0.88, 0.47, 0], [0.7, 0, 0.71]], np.float32)
    )
    graph = HNSWGraph(vectors[:1], m=2)
    graph._v = vectors
    ranked = sorted(((float(vectors[i] @ vectors[0]), i) for i in (1, 2, 3)), reverse=True)
    # b (2) is closer to a (1) than to the base, so it is skipped for c (3)
    assert graph._select(ranked, 2) == [1, 3]


def test_edge_cases() -> None:
    small = HNSWGraph(VECTORS[:3], m=2)
    assert len(small.search(QUERIES[0], 10, ef=5)) == 3  # k > n returns all
    with pytest.raises(ValueError, match="k must be"):
        small.search(QUERIES[0], 0, ef=5)
    with pytest.raises(ValueError, match="m must be"):
        HNSWGraph(VECTORS, m=1)
    assert HNSWGraph(np.zeros((0, 32), np.float32)).search(QUERIES[0], 3, ef=5) == []


def _exact_store(n: int) -> ExactVectorStore:
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="D",
            text=f"t{i}",
            char_start=0,
            char_end=2,
            page_start=0,
            page_end=0,
            metadata={"company": "A" if i % 10 == 0 else "B"},
        )
        for i in range(n)
    ]
    store = ExactVectorStore("fake", VECTORS.shape[1])
    store.upsert(chunks, VECTORS[:n])
    return store


def test_store_uses_exact_search_for_small_filtered_sets() -> None:
    exact = _exact_store(600)
    store = HNSWVectorStore(exact, m=8, ef_search=32, exact_below=100)
    q = QUERIES[0]
    # 60 chunks of company A <= 100: exact, identical to the exact store
    assert store.search(q, 5, {"company": "A"}) == exact.search(q, 5, {"company": "A"})
    assert store.exact_fallbacks == 1
    calls: list[Filters] = []
    original = exact.filter_mask

    def counting(filters: Filters) -> npt.NDArray[np.bool_]:
        calls.append(filters)
        return original(filters)

    exact.filter_mask = counting  # type: ignore[method-assign]
    store.search(q, 5, {"company": "A"})
    assert len(calls) == 1  # the mask is built once and reused by the exact fallback
    exact.filter_mask = original  # type: ignore[method-assign]
    hits = store.search(q, 5, {"company": "B"})  # 540 chunks: the graph, filtered
    assert store.exact_fallbacks == 2
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.chunk.metadata["company"] == "B" for h in hits)
    unfiltered = store.search(q, 5)
    assert {h.chunk.chunk_id for h in unfiltered} == {h.chunk.chunk_id for h in exact.search(q, 5)}
    assert store.chunks == exact.chunks and len(store) == 600
    with pytest.raises(NotImplementedError):
        store.upsert([], VECTORS[:0])


def test_config_selects_the_index() -> None:
    exact = _exact_store(50)
    assert DenseIndexConfig().kind is DenseIndexKind.EXACT
    assert dense_search(DenseIndexConfig(), exact) is exact
    hnsw = dense_search(DenseIndexConfig(kind=DenseIndexKind.HNSW, m=4), exact)
    assert isinstance(hnsw, HNSWVectorStore) and hnsw.model_id == "fake"


def test_exact_store_vectors_are_read_only() -> None:
    vectors = _exact_store(5).vectors
    with pytest.raises(ValueError, match="read-only"):
        vectors[0, 0] = 1.0
