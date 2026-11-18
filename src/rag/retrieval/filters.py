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


def normalize(text: str) -> str:
    """Lowercase, non-alphanumerics → single spaces, padded so ' alias ' matches whole words."""
    return f" {_NON_ALNUM.sub(' ', text.lower()).strip()} "


class CatalogFilterExtractor(FilterExtractor):
    """Matches known company aliases and explicit years in the question text.

    - Company: an alias must appear as whole words. If aliases of two different
      companies appear, no company filter is returned: a wrong hard filter hides the
      answer completely, which is worse than not filtering.
    - Year: every mentioned year, widened by ``year_window`` on each side
      (forward-looking questions are answered by the previous year's filing).
    """

    def __init__(
        self,
        company_aliases: Mapping[str, Iterable[str]],
        year_window: int = 0,
        extract_company: bool = True,
        extract_year: bool = True,
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
        window = range(-self._year_window, self._year_window + 1)
        return tuple(sorted({year + offset for year in mentioned for offset in window}))
