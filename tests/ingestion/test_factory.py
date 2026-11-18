from rag.config import ChunkingConfig, ParsingConfig
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.factory import build_chunker, build_parser
from rag.ingestion.pdf_parser import PypdfParser
from support.tokenizers import WordCounter


def test_default_config_builds_pypdf_parser() -> None:
    assert isinstance(build_parser(ParsingConfig()), PypdfParser)


def test_builds_recursive_chunker_from_config() -> None:
    chunker = build_chunker(ChunkingConfig(chunk_size=50, chunk_overlap=5), WordCounter())
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.fingerprint == RecursiveChunker(WordCounter(), 50, 5).fingerprint
