"""Config → embedder, and where its index lives."""

from __future__ import annotations

import re
from pathlib import Path

from rag.config import AppConfig, EmbeddingConfig
from rag.indexing.embedders import SentenceTransformerEmbedder


def build_embedder(config: EmbeddingConfig) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=config.model_id,
        revision=config.revision,
        query_prefix=config.query_prefix,
        batch_size=config.batch_size,
        device=config.device,
    )


def model_slug(model_id: str) -> str:
    """File-system-safe model name, e.g. ``BAAI/bge-small-en-v1.5`` → ``baai-bge-small-en-v1-5``."""
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def dense_index_dir(config: AppConfig, chunker_fingerprint: str) -> Path:
    """Keyed by chunker *and* model, so neither change can silently reuse a stale index."""
    name = f"{chunker_fingerprint}__{model_slug(config.embedding.model_id)}"
    return config.paths.data_dir / "indexes" / "dense" / name
