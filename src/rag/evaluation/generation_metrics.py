"""Deterministic answer checks (HLD 010). LLM-judge metrics come in backlog 2.2.

These are cheap and reproducible, but crude: they catch format, citation and grounding
failures, and give a rough correctness signal for numeric answers only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from rag.domain import Chunk, GoldenItem
from rag.evaluation.evidence import covers

_NUMERIC_REFERENCE = re.compile(r"\s*\$?\s*(-?\(?[\d,]*\.?\d+\)?)\s*%?\s*")
_NUMBER = re.compile(r"(?<![\w.])-?\(?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?%?")


def _to_float(raw: str) -> float:
    negative = raw.startswith("-") or (raw.startswith("(") and raw.endswith(")"))
    value = float(raw.strip("-()").replace(",", ""))
    return -value if negative else value


def reference_number(reference_answer: str) -> float | None:
    """The value of a reference answer that is *only* a number ("$1577.00", "8.70", "-3%")."""
    match = _NUMERIC_REFERENCE.fullmatch(reference_answer)
    return _to_float(match.group(1)) if match else None


def numbers_in(text: str) -> list[float]:
    """Every number in free text, with thousands separators removed ("1,577" → 1577)."""
    return [
        float((m.group(1) + (m.group(2) or "")).replace(",", "")) for m in _NUMBER.finditer(text)
    ]


def numeric_match(answer: str, reference: float, rel_tol: float = 0.01) -> bool:
    """Does the answer state the reference value (sign-insensitive, within ``rel_tol``)?"""
    target = abs(reference)
    return any(
        math.isclose(value, target, rel_tol=rel_tol, abs_tol=0.005) for value in numbers_in(answer)
    )


def cites_evidence(cited: Iterable[Chunk], item: GoldenItem) -> bool:
    """At least one cited chunk covers a labelled evidence page (page-level, like retrieval)."""
    return any(covers(chunk, span) for chunk in cited for span in item.evidence)
