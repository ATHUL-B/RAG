"""Rule-based extraction of company and year filters from a question (HLD 006).

Deterministic and free: the baseline that an LLM-based query planner (Part 7) will
later have to beat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final

from rag.domain import Filters
from rag.retrieval.base import FilterExtractor

COMPANY_KEY: Final = "company"
YEAR_KEY: Final = "year"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"(?<!\d)(?<!\d[,.])(199\d|20[0-4]\d)(?!\d|[,.]\d)")
"""A 4-digit year 1990-2049, not part of a longer number (12,019 / 2019.5 / 2,019,000).

A comma or period *followed by a digit* means a number; followed by a space it is prose
("FY2023, FY2022 and FY2021"), so the year still counts.
"""
_SHORT_FISCAL_YEAR = re.compile(r"\bFY\s?'?(\d{2})(?![\d,.]\d|\d)", re.IGNORECASE)
"""Two-digit fiscal years: "FY22", "FY 20", "FY'21". Only with the FY prefix: a bare "22"
is far more often a count or a percentage than a year."""
_FORWARD_LOOKING = re.compile(
    r"\b(?:expect(?:s|ed|ing|ation|ations)?|forecast(?:s|ed|ing)?|guidance|outlook"
    r"|anticipat(?:e|es|ed|ing)|project(?:s|ed|ion|ions)?)\b",
    re.IGNORECASE,
)
"""Cue words for a forecast. A forecast *for* FY2023 is written in the FY2022 filing."""


def normalize(text: str) -> str:
    """Lowercase, non-alphanumerics → single spaces, padded so ' alias ' matches whole words."""
    return f" {_NON_ALNUM.sub(' ', text.lower()).strip()} "


class CatalogFilterExtractor(FilterExtractor):
    """Matches known company aliases and explicit years in the question text.

    - Company: an alias must appear as whole words. If aliases of two different
      companies appear, no company filter is returned: a wrong hard filter hides the
      answer completely, which is worse than not filtering.
    - Year: every mentioned year, widened by ``year_window`` on each side.
    - ``short_fiscal_years``: also read "FY22" as 2022 (rule-based planner, backlog 3.6).
    - ``forward_looking_prior_year``: when the question asks about a forecast ("expected
      in FY2023"), also allow the year before each mentioned year, because the forecast is
      published in the previous year's filing. Unlike ``year_window`` this widens only
      forward-looking questions (a blanket +-1 window was measured worse in HLD 006).
    """

    def __init__(
        self,
        company_aliases: Mapping[str, Iterable[str]],
        year_window: int = 0,
        extract_company: bool = True,
        extract_year: bool = True,
        short_fiscal_years: bool = False,
        forward_looking_prior_year: bool = False,
    ) -> None:
        if year_window < 0:
            raise ValueError("year_window must be >= 0")
        self._aliases: list[tuple[str, str]] = sorted(
            {
                (normalize(alias), company)
                for company, aliases in company_aliases.items()
                for alias in (company, *aliases)
                if normalize(alias).strip()
            },
            key=lambda pair: -len(pair[0]),  # longest alias first, for readability of matches
        )
        self._year_window = year_window
        self._extract_company = extract_company
        self._extract_year = extract_year
        self._short_fiscal_years = short_fiscal_years
        self._forward_looking_prior_year = forward_looking_prior_year

    def extract(self, text: str) -> Filters:
        filters: Filters = {}
        if self._extract_company and (company := self._company(text)) is not None:
            filters[COMPANY_KEY] = company
        if self._extract_year and (years := self._years(text)):
            filters[YEAR_KEY] = years
        return filters

    def _company(self, text: str) -> str | None:
        normalized = normalize(text)
        found = {company for alias, company in self._aliases if alias in normalized}
        return found.pop() if len(found) == 1 else None

    def _years(self, text: str) -> tuple[int, ...]:
        mentioned = {int(y) for y in _YEAR.findall(text)}
        if self._short_fiscal_years:
            mentioned |= {_century(int(yy)) for yy in _SHORT_FISCAL_YEAR.findall(text)}
        window = range(-self._year_window, self._year_window + 1)
        years = {year + offset for year in mentioned for offset in window}
        if self._forward_looking_prior_year and _FORWARD_LOOKING.search(text):
            years |= {year - 1 for year in mentioned}
        return tuple(sorted(years))


def _century(two_digits: int) -> int:
    """FY95 → 1995, FY22 → 2022 (the corpus spans the 1990s to the 2040s at most)."""
    return 1900 + two_digits if two_digits >= 90 else 2000 + two_digits
