"""Deterministic identifiers.

Same input -> same ID, on any machine, on any run. This is what makes it possible
to diff two ingestion runs, cache embeddings, and trace an answer back to the
exact text that was indexed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_ID_HEX_LENGTH = 16  # 64 bits: collision-safe far beyond our corpus sizes


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable short hash of a config mapping (key order does not matter)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_hex(canonical)[:_ID_HEX_LENGTH]


def make_chunk_id(doc_id: str, char_start: int, char_end: int, chunker_fingerprint: str) -> str:
    """ID for a chunk: changes if the document, the span, or the chunker config changes."""
    if char_end <= char_start:
        raise ValueError("char_end must be greater than char_start")
    return _sha256_hex(f"{doc_id}|{char_start}|{char_end}|{chunker_fingerprint}")[:_ID_HEX_LENGTH]
