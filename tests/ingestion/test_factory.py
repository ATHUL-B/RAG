from rag.config import ParsingConfig
from rag.ingestion.factory import build_parser
from rag.ingestion.pdf_parser import PypdfParser


def test_default_config_builds_pypdf_parser() -> None:
    assert isinstance(build_parser(ParsingConfig()), PypdfParser)
