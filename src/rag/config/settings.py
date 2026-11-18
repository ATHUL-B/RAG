"""Typed, validated configuration.

Every tunable number (chunk size, top-K, ...) lives here and in ``configs/*.yaml``,
never as a literal in code, so an experiment is fully described by its config
file plus the git SHA (CLAUDE.md §2). Sections are added as components land.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator


class StrictConfig(BaseModel):
    """Unknown keys are errors: a typo in YAML must fail loudly, not be ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetName(StrEnum):
    FINANCEBENCH = "financebench"


class ParserName(StrEnum):
    PYPDF = "pypdf"


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"


class PathsConfig(StrictConfig):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def parsed_dir(self) -> Path:
        """Parsed ``Document`` artifacts, one JSON per document."""
        return self.processed_dir / "parsed"

    def chunks_dir(self, chunker_fingerprint: str) -> Path:
        """Chunk artifacts for one chunker configuration (configs coexist for ablations)."""
        return self.processed_dir / "chunks" / chunker_fingerprint


class ParsingConfig(StrictConfig):
    parser: ParserName = ParserName.PYPDF


class ChunkingConfig(StrictConfig):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: PositiveInt = 400
    chunk_overlap: NonNegativeInt = 0
    tokenizer: str = Field(default="cl100k_base", min_length=1)
    """tiktoken encoding used to measure chunk_size and chunk_overlap."""

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(StrictConfig):
    """Defaults follow ADR 0003."""

    model_id: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    revision: str | None = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    """Pinned model revision: the same config always means the same weights."""
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: PositiveInt = 64
    device: str = Field(default="auto", pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$")


class RetrievalConfig(StrictConfig):
    top_k: PositiveInt = 5


class EvaluationConfig(StrictConfig):
    ks: tuple[PositiveInt, ...] = Field(default=(1, 3, 5, 10), min_length=1)
    """Cut-offs for recall@K, precision@K, hit@K and nDCG@K."""
    parse_audit_min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    """Evidence passes the parse audit when this share of its tokens is on the parsed page."""


class AppConfig(StrictConfig):
    experiment_name: str = Field(min_length=1)
    dataset: DatasetName = DatasetName.FINANCEBENCH
    paths: PathsConfig = PathsConfig()
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
