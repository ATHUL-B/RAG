"""Rule-based query rewriting: clean the search text before embedding it (Part 7; HLD 011).

FinanceBench questions wrap a short question in instructions: a persona ("Assume you are
an equity analyst"), formatting ("Round your answer to two decimal places") and caveats
("If operating margin is not a useful metric ... explain why"). Embedded as-is, much of
the vector describes the instructions, not the fact we are looking for.

``RuleBasedRewriter`` makes one cleaner search text from the question:

1. **Drop instruction sentences** (persona, rounding, units, "explain why" caveats). A
   lead-in sentence that *contains* the question ("Based on the balance sheet, what is
   ...?") keeps the question part.
2. **Keep what helps retrieval:** the core question, definitions ("DPO is defined as:
   365 * average accounts payable ...") and the statements named, rewritten to the titles
   the filings use ("P&L statement" → "consolidated statements of income operations").
3. **Expand abbreviations** from a small glossary ("PP&E" → "property, plant and
   equipment"). Only spelled-out names of the abbreviation itself: the rewriter must not
   add facts the question does not contain (Part 7, rule 2).

This is the free baseline an LLM planner has to beat. It never sees the corpus, only the
question text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from rag.domain import Query
from rag.retrieval.base import QueryRewriter

_SENTENCE: Final = re.compile(r"(?<=[.?!])\s+(?=[A-Z])|;\s*")
"""Sentence breaks, and ';' so "…FY2022; if X is not meaningful, explain why" splits."""
_PARENTHETICAL_INSTRUCTION: Final = re.compile(
    r"\s*\([^()]*\b(?:round|units of|in usd)\b[^()]*\)", re.IGNORECASE
)
"""Formatting asides inside a question: "(in units of percents and round to one decimal)"."""

_INSTRUCTION: Final = re.compile(
    r"\b(?:round(?:ed)? (?:your |the )?(?:answer )?to|answer in (?:usd|units)"
    r"|explain why|then state 0|state that and|you are an?\b|assum(?:e|ing)\b"
    r"|perspective of|standpoint of|as if you|we need to calculate|base your judgments?"
    r"|basing your judgments?|give a (?:response|solution)|calculate what was asked"
    r"|calculate \(or extract\)|address the question|respond to the question"
    r"|provide a response|approach the question|answer the question asked"
    r"|please utilize|answer the following question|here'?s the question"
    r"|based on the information|by drawing conclusions|in agreement with"
    r"|considering the data|taking into account|according to the details"
    r"|using the (?:cash flow|income|balance|statement|p&l))",
    re.IGNORECASE,
)
"""Phrases that mark a sentence as an instruction rather than the question itself."""

_QUESTION_TAIL: Final = re.compile(
    r"[:,]\s*((?:what|how|which|who|when|where|why|does|did|do|is|are|was|were|has|have"
    r"|roughly)\b.*)$",
    re.IGNORECASE,
)
"""Inside an instruction sentence, the question after a ':' or ',' lead-in."""

STATEMENT_TITLES: Final[Mapping[str, str]] = {
    r"balance sheets?|statements? of financial position": "consolidated balance sheets",
    r"income statements?|statements? of income|p&l(?: statement)?|statements? of operations": (
        "consolidated statements of income operations"
    ),
    r"cash flow statements?|statements? of cash flows?": "consolidated statements of cash flows",
}
"""Ways questions name a financial statement → the title the filings use."""

ABBREVIATIONS: Final[Mapping[str, str]] = {
    r"PP&?N?E": "property, plant and equipment net",
    r"capex": "capital expenditures purchases of property, plant and equipment",
    r"COGS": "cost of goods sold cost of sales",
    r"D&A": "depreciation and amortization",
    r"SG&A": "selling, general and administrative expenses",
    r"EBITDA": "earnings before interest, taxes, depreciation and amortization",
    r"EPS": "earnings per share",
    r"DPO": "days payable outstanding",
    r"DSO": "days sales outstanding",
    r"DIO": "days inventory outstanding",
    r"CCC": "cash conversion cycle",
    r"ROA": "return on assets",
    r"ROE": "return on equity",
    r"FCF": "free cash flow",
    r"CAGR": "compound annual growth rate",
    r"AGM": "annual meeting of shareholders",
}
"""Abbreviation (regex, whole word) → its spelled-out name."""


def _whole_word(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", re.IGNORECASE)


_STATEMENTS: Final = [(_whole_word(p), title) for p, title in STATEMENT_TITLES.items()]
_ABBREVIATIONS: Final = [(_whole_word(p), name) for p, name in ABBREVIATIONS.items()]


def strip_boilerplate(question: str) -> str:
    """The question without instruction sentences; statement mentions become titles."""
    kept: list[str] = []
    text = _PARENTHETICAL_INSTRUCTION.sub("", question.strip())
    for sentence in _SENTENCE.split(text):
        if not _INSTRUCTION.search(sentence):
            kept.append(sentence)
        elif tail := _QUESTION_TAIL.search(sentence):
            kept.append(tail.group(1))
    if not kept:  # never lose the question: an over-eager rule falls back to the original
        kept = [question.strip()]
    titles = [title for pattern, title in _STATEMENTS if pattern.search(question)]
    return " ".join([*kept, *dict.fromkeys(titles)]).strip()


def expand_abbreviations(text: str) -> str:
    """Append the spelled-out name of every abbreviation found (once each)."""
    names = [name for pattern, name in _ABBREVIATIONS if pattern.search(text)]
    return " ".join([text, *dict.fromkeys(names)]) if names else text


class RuleBasedRewriter(QueryRewriter):
    """Returns at most one rewritten query; none when the rules change nothing."""

    def __init__(self, strip: bool = True, expand: bool = True) -> None:
        if not (strip or expand):
            raise ValueError("enable at least one rewrite rule")
        self._strip = strip
        self._expand = expand

    def rewrite(self, query: Query) -> list[Query]:
        text = strip_boilerplate(query.text) if self._strip else query.text
        text = expand_abbreviations(text) if self._expand else text
        if not text or text == query.text:
            return []
        return [query.model_copy(update={"text": text})]
