from rag.config import ChunkingConfig, ContextConfig, ParsingConfig
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.context import ContextualChunker
from rag.ingestion.factory import build_chunker, build_parser
from rag.ingestion.pdf_parser import PypdfParser
from support.tokenizers import WordCounter


def test_default_config_builds_pypdf_parser() -> None:
    assert isinstance(build_parser(ParsingConfig()), PypdfParser)


def test_builds_recursive_chunker_from_config() -> None:
    chunker = build_chunker(ChunkingConfig(chunk_size=50, chunk_overlap=5), WordCounter())
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.fingerprint == RecursiveChunker(WordCounter(), 50, 5).fingerprint


def test_context_wraps_the_chunker() -> None:
    config = ChunkingConfig(respect_pages=True, context=ContextConfig(enabled=True))
    chunker = build_chunker(config, WordCounter())
    assert isinstance(chunker, ContextualChunker)
    plain = build_chunker(config.model_copy(update={"context": ContextConfig()}), WordCounter())
    assert chunker.fingerprint != plain.fingerprint
