"""Config → concrete implementation. Ablations change the config, not the code."""

from __future__ import annotations

from rag.config import ChunkingConfig, ChunkingStrategy, ParserName, ParsingConfig
from rag.ingestion.base import Chunker, Parser
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.context import ContextualChunker, HeaderContextualizer
from rag.ingestion.pdf_parser import PypdfLayoutParser, PypdfParser
from rag.ingestion.tokenization import TiktokenCounter, TokenCounter


def build_parser(config: ParsingConfig) -> Parser:
    match config.parser:
        case ParserName.PYPDF:
            return PypdfParser()
        case ParserName.PYPDF_LAYOUT:
            return PypdfLayoutParser()


def build_chunker(
    config: ChunkingConfig,
    token_counter: TokenCounter | None = None,
    parser: ParserName = ParserName.PYPDF,
) -> Chunker:
    """``parser`` names the text being chunked: chunk IDs (char offsets) and the index only
    make sense for that parser's text, so it is part of the fingerprint (HLD 021)."""
    counter = token_counter or TiktokenCounter(config.tokenizer)
    chunker: Chunker
    match config.strategy:
        case ChunkingStrategy.RECURSIVE:
            chunker = RecursiveChunker(
                counter,
                config.chunk_size,
                config.chunk_overlap,
                respect_pages=config.respect_pages,
                source=str(parser),
            )
    if config.context.enabled:
        contextualizer = HeaderContextualizer(
            document_fields=config.context.document_fields,
            page_title=config.context.page_title,
            max_title_lines=config.context.max_title_lines,
        )
        chunker = ContextualChunker(chunker, contextualizer)
    return chunker
