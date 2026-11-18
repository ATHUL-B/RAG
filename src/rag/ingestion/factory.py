"""Config → concrete implementation. Ablations change the config, not the code."""

from __future__ import annotations

from rag.config import ChunkingConfig, ChunkingStrategy, ParserName, ParsingConfig
from rag.ingestion.base import Chunker, Parser
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.pdf_parser import PypdfParser
from rag.ingestion.tokenization import TiktokenCounter, TokenCounter


def build_parser(config: ParsingConfig) -> Parser:
    match config.parser:
        case ParserName.PYPDF:
            return PypdfParser()


def build_chunker(config: ChunkingConfig, token_counter: TokenCounter | None = None) -> Chunker:
    counter = token_counter or TiktokenCounter(config.tokenizer)
    match config.strategy:
        case ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(counter, config.chunk_size, config.chunk_overlap)
