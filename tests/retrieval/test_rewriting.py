import pytest

from rag.domain import Query
from rag.retrieval.base import QueryRewriter
from rag.retrieval.rewriting import (
    CompositeRewriter,
    RuleBasedRewriter,
    expand_abbreviations,
    strip_boilerplate,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # instruction sentences are dropped
        (
            "Does Adobe have an improving operating margin profile as of FY2022? If operating "
            "margin is not a useful metric for a company like this, then state that and "
            "explain why.",
            "Does Adobe have an improving operating margin profile as of FY2022?",
        ),
        # a lead-in sentence keeps its question; the statement becomes its filing title
        (
            "Considering the data in the balance sheet, what is Block's FY2016 working capital "
            "ratio? Round your answer to two decimal places.",
            "what is Block's FY2016 working capital ratio? consolidated balance sheets",
        ),
        # definitions are kept (they name the line items); formatting asides are removed
        (
            "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units "
            "of percents and round to one decimal place)? Calculate what was asked by utilizing "
            "the line items clearly shown in the statement of income.",
            "What is Amazon's year-over-year change in revenue from FY2016 to FY2017? "
            "consolidated statements of income operations",
        ),
        # ';' separates a useful clause from a caveat
        (
            "Roughly how many times has JnJ sold its inventory in FY2022? Calculate inventory "
            "turnover ratio for FY2022; if conventional inventory management is not meaningful "
            "for the company then state that and explain why.",
            "Roughly how many times has JnJ sold its inventory in FY2022? Calculate inventory "
            "turnover ratio for FY2022",
        ),
        # "Here's the question:" lead-in; two statements named, each title once
        (
            "You are an investment banker and your only resource(s) to answer the following "
            "question is (are): the statement of financial position and the cash flow statement. "
            "Here's the question: what is the FY2015 operating cash flow ratio for Adobe? "
            "Operating cash flow ratio is defined as: cash from operations / total current "
            "liabilities.",
            "what is the FY2015 operating cash flow ratio for Adobe? Operating cash flow ratio "
            "is defined as: cash from operations / total current liabilities. consolidated "
            "balance sheets consolidated statements of cash flows",
        ),
        ("What was 3M's FY2018 capex?", "What was 3M's FY2018 capex?"),  # nothing to strip
    ],
)
def test_strip_boilerplate(question: str, expected: str) -> None:
    assert strip_boilerplate(question) == expected


def test_strip_never_returns_an_empty_question() -> None:
    question = "Round your answer to two decimal places."
    assert strip_boilerplate(question) == question


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Did Pfizer grow its PPNE?",
            "Did Pfizer grow its PPNE? property, plant and equipment net",
        ),
        (
            "PP&E and COGS",
            "PP&E and COGS property, plant and equipment net cost of goods sold cost of sales",
        ),
        ("ppe ppe", "ppe ppe property, plant and equipment net"),  # once, case-insensitive
        ("The EPSILON fund", "The EPSILON fund"),  # whole words only
        ("No abbreviations here", "No abbreviations here"),
    ],
)
def test_expand_abbreviations(text: str, expected: str) -> None:
    assert expand_abbreviations(text) == expected


def test_rewriter_keeps_filters_and_returns_nothing_when_unchanged() -> None:
    rewriter = RuleBasedRewriter(strip=True, expand=True)
    query = Query(
        text="FY2018 capex? Round your answer to two decimal places.", filters={"year": 2018}
    )
    (rewritten,) = rewriter.rewrite(query)
    assert rewritten.text == (
        "FY2018 capex? capital expenditures purchases of property, plant and equipment"
    )
    assert rewritten.filters == {"year": 2018}
    assert rewriter.rewrite(Query(text="What was revenue?")) == []


def test_rules_can_be_used_alone_but_not_both_off() -> None:
    query = Query(text="PP&E? Round your answer to two decimal places.")
    assert RuleBasedRewriter(strip=True, expand=False).rewrite(query)[0].text == "PP&E?"
    assert RuleBasedRewriter(strip=False, expand=True).rewrite(query)[0].text.endswith("net")
    with pytest.raises(ValueError, match="at least one"):
        RuleBasedRewriter(strip=False, expand=False)


class Fixed(QueryRewriter):
    def __init__(self, *texts: str) -> None:
        self._texts = texts

    def rewrite(self, query: Query) -> list[Query]:
        return [query.model_copy(update={"text": t}) for t in self._texts]


def test_composite_runs_rewriters_in_order_without_duplicates() -> None:
    composite = CompositeRewriter([Fixed("clean", "q"), Fixed("clean", "sub 1", "sub 2")])
    assert [r.text for r in composite.rewrite(Query(text="q"))] == ["clean", "sub 1", "sub 2"]
    with pytest.raises(ValueError, match="at least one"):
        CompositeRewriter([])
