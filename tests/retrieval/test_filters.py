import pytest

from rag.retrieval.filters import CatalogFilterExtractor, normalize

ALIASES = {
    "Johnson & Johnson": ("JnJ", "J&J"),
    "Coca-Cola": ("Coca Cola",),
    "3M": ("MMM",),
    "AMD": (),
    "American Express": ("AmEx",),
}


def _extract(text: str, **kwargs: int | bool) -> dict[str, object]:
    return dict(CatalogFilterExtractor(ALIASES, **kwargs).extract(text))  # type: ignore[arg-type]


class TestCompany:
    @pytest.mark.parametrize(
        ("question", "company"),
        [
            ("What was JnJ's FY2022 revenue?", "Johnson & Johnson"),
            ("Did J&J grow?", "Johnson & Johnson"),
            ("Coca Cola dividend payout", "Coca-Cola"),
            ("coca-cola's margin", "Coca-Cola"),
            ("3M capex", "3M"),
            ("Is AMEX improving?", "American Express"),
        ],
    )
    def test_matches_names_and_aliases_case_and_punctuation_insensitively(
        self, question: str, company: str
    ) -> None:
        assert _extract(question)["company"] == company

    def test_alias_must_be_whole_words(self) -> None:
        assert "company" not in _extract("The AMDOCS contract")  # not AMD
        assert "company" not in _extract("MMMM")

    def test_two_different_companies_means_no_company_filter(self) -> None:
        assert "company" not in _extract("Compare AMD and 3M revenue")

    def test_two_aliases_of_the_same_company_are_fine(self) -> None:
        assert _extract("JnJ (Johnson & Johnson) sales")["company"] == "Johnson & Johnson"


class TestYears:
    @pytest.mark.parametrize(
        ("question", "years"),
        [
            ("FY2018 capex", (2018,)),
            ("as of FY 2021, how much", (2021,)),
            ("FY2023, FY2022 and FY2021 acquisitions", (2021, 2022, 2023)),
            ("between FY 2023 and Q2 of FY2024?", (2023, 2024)),
            ("in fiscal year 1999.", (1999,)),
        ],
    )
    def test_finds_every_mentioned_year(self, question: str, years: tuple[int, ...]) -> None:
        assert _extract(question)["year"] == years

    @pytest.mark.parametrize(
        "text", ["revenue of 12,019 million", "ratio 2019.5", "$2,019,000", "1850"]
    )
    def test_numbers_are_not_years(self, text: str) -> None:
        assert "year" not in _extract(text)

    def test_year_window_widens_each_year(self) -> None:
        assert _extract("FY2018 vs FY2020", year_window=1)["year"] == (2017, 2018, 2019, 2020, 2021)

    def test_rejects_negative_window(self) -> None:
        with pytest.raises(ValueError, match="year_window"):
            CatalogFilterExtractor(ALIASES, year_window=-1)


def test_fields_can_be_switched_off() -> None:
    question = "3M FY2018 capex"
    assert _extract(question, extract_company=False) == {"year": (2018,)}
    assert _extract(question, extract_year=False) == {"company": "3M"}


def test_question_without_company_or_year_has_no_filters() -> None:
    assert _extract("What drove the reduction in SG&A expense?") == {}


def test_normalize_pads_and_collapses_punctuation() -> None:
    assert normalize("Coca-Cola's  FY2022!") == " coca cola s fy2022 "


class TestShortFiscalYears:
    @pytest.mark.parametrize(
        ("question", "years"),
        [
            ("What drove revenue change as of the FY22 for AMD?", (2022,)),
            ("Did Pfizer grow its PPNE between FY20 and FY21?", (2020, 2021)),
            ("fy 19 margin", (2019,)),
            ("FY'21 capex", (2021,)),
            ("FY99 results", (1999,)),
            ("FY2023Q1 guidance", (2023,)),  # a 4-digit year is not also read as FY20
            ("as of FY 2021", (2021,)),
        ],
    )
    def test_reads_two_digit_fiscal_years(self, question: str, years: tuple[int, ...]) -> None:
        assert _extract(question, short_fiscal_years=True)["year"] == years

    @pytest.mark.parametrize("text", ["revenue grew 22%", "FY22.5 units", "FY221", "fy 22,5"])
    def test_requires_the_fy_prefix_and_a_whole_number(self, text: str) -> None:
        assert "year" not in _extract(text, short_fiscal_years=True)

    def test_off_by_default(self) -> None:
        assert "year" not in _extract("as of the FY22")


class TestForwardLooking:
    @pytest.mark.parametrize(
        ("question", "years"),
        [
            ("Is growth in JnJ's adjusted EPS expected to accelerate in FY2023?", (2022, 2023)),
            ("What production rate changes is Boeing forecasting for FY2023?", (2022, 2023)),
            ("FY2024 guidance and outlook", (2023, 2024)),
            ("What was FY2022 revenue?", (2022,)),  # not a forecast: unchanged
        ],
    )
    def test_forecasts_also_allow_the_previous_years_filing(
        self, question: str, years: tuple[int, ...]
    ) -> None:
        assert _extract(question, forward_looking_prior_year=True)["year"] == years

    def test_does_not_invent_a_year(self) -> None:
        assert "year" not in _extract(
            "How much does Pfizer expect to pay?", forward_looking_prior_year=True
        )

    def test_combines_with_short_fiscal_years(self) -> None:
        years = _extract(
            "AMD outlook for FY23", short_fiscal_years=True, forward_looking_prior_year=True
        )
        assert years["year"] == (2022, 2023)
