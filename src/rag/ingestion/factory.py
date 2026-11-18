"""Config → concrete implementation. Ablations change the config, not the code."""

from __future__ import annotations

from rag.config import ChunkingConfig, ChunkingStrategy, ParserName, ParsingConfig
from rag.ingestion.base import Chunker, Parser
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.context import ContextualChunker, HeaderContextualizer
from rag.ingestion.pdf_parser import PypdfParser
from rag.ingestion.tokenization import TiktokenCounter, TokenCounter


def build_parser(config: ParsingConfig) -> Parser:
    match config.parser:
        case ParserName.PYPDF:
            return PypdfParser()


def build_chunker(config: ChunkingConfig, token_counter: TokenCounter | None = None) -> Chunker:
    counter = token_counter or TiktokenCounter(config.tokenizer)
    chunker: Chunker
    match config.strategy:
        case ChunkingStrategy.RECURSIVE:
            chunker = RecursiveChunker(
                counter,
                config.chunk_size,
                config.chunk_overlap,
                respect_pages=config.respect_pages,
            )
    if config.context.enabled:
        contextualizer = HeaderContextualizer(
            document_fields=config.context.document_fields,
            page_title=config.context.page_title,
            max_title_lines=config.context.max_title_lines,
        )
        chunker = ContextualChunker(chunker, contextualizer)
    return chunker
