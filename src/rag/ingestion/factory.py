"""Config → concrete implementation. Ablations change the config, not the code."""

from __future__ import annotations

from rag.config import ParserName, ParsingConfig
from rag.ingestion.base import Parser
from rag.ingestion.pdf_parser import PypdfParser


def build_parser(config: ParsingConfig) -> Parser:
    match config.parser:
        case ParserName.PYPDF:
            return PypdfParser()
