from rag.domain import Chunk, Query
from rag.indexing.vector_store import ExactVectorStore
from rag.retrieval.dense import DenseRetriever
from support.embedders import HashingEmbedder


def _chunk(chunk_id: str, text: str, year: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"D_{year}",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=0,
        page_end=0,
        metadata={"year": year},
    )


def _retriever() -> DenseRetriever:
    embedder = HashingEmbedder()
    store = ExactVectorStore(embedder.model_id, embedder.dimension)
    chunks = [
        _chunk("a", "net income rose sharply", 2023),
        _chunk("b", "net income fell", 2022),
        _chunk("c", "board of directors", 2023),
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return DenseRetriever(embedder, store)


def test_retrieves_top_k_by_similarity() -> None:
    hits = _retriever().retrieve(Query(text="net income"), k=2)
    assert {h.chunk.chunk_id for h in hits} == {"a", "b"}
    assert [h.rank for h in hits] == [1, 2]


def test_passes_query_filters_to_the_store() -> None:
    hits = _retriever().retrieve(Query(text="net income", filters={"year": 2022}), k=5)
    assert [h.chunk.chunk_id for h in hits] == ["b"]
