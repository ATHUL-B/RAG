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


def test_layout_parser_is_selectable() -> None:
    from rag.config import ParserName
    from rag.ingestion.pdf_parser import PypdfLayoutParser

    assert isinstance(
        build_parser(ParsingConfig(parser=ParserName.PYPDF_LAYOUT)), PypdfLayoutParser
    )


def test_parser_is_part_of_the_fingerprint_but_the_baseline_keeps_its_own() -> None:
    from rag.config import ParserName

    config = ChunkingConfig(respect_pages=True, context=ContextConfig(enabled=True))
    default = build_chunker(config, WordCounter())
    baseline = build_chunker(config, WordCounter(), parser=ParserName.PYPDF)
    layout = build_chunker(config, WordCounter(), parser=ParserName.PYPDF_LAYOUT)
    assert default.fingerprint == baseline.fingerprint  # existing indexes stay valid
    assert layout.fingerprint != baseline.fingerprint  # different text, different chunk IDs


def test_parsed_artifacts_are_kept_per_parser() -> None:
    from rag.config.settings import PathsConfig

    paths = PathsConfig()
    assert paths.parsed_dir_for("pypdf") == paths.parsed_dir
    assert paths.parsed_dir_for("pypdf-layout") == paths.processed_dir / "parsed-pypdf-layout"
