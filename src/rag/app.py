"""Composition root: build the configured retrieval stack from artifacts on disk.

Scripts call this instead of wiring components themselves, so every entry point searches
exactly the same way for a given config.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import AppConfig, RetrievalMode
from rag.datasets.financebench import COMPANY_ALIASES
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.retrieval.base import Retriever
from rag.retrieval.factory import build_bm25_index, build_retriever


@dataclass(frozen=True)
class RetrievalStack:
    retriever: Retriever
    store: ExactVectorStore


def load_retrieval_stack(config: AppConfig) -> RetrievalStack:
    """Load the dense index for this config's chunker + model, then compose the retriever."""
    embedder = build_embedder(config.embedding)
    index_dir = dense_index_dir(config, build_chunker(config.chunking).fingerprint)
    store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
    keyword_index = (
        build_bm25_index(config.retrieval.bm25, store.chunks)
        if config.retrieval.mode is not RetrievalMode.DENSE
        else None
    )
    retriever = build_retriever(config.retrieval, embedder, store, COMPANY_ALIASES, keyword_index)
    embedder.embed_query("warm-up")  # load the model before any timed work
    return RetrievalStack(retriever=retriever, store=store)
