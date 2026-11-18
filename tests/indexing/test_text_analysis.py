import pytest

from rag.indexing.text_analysis import analyze


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Purchases of Property, Plant", ["purchases", "property", "plant"]),
        ("capex was $1,577 million", ["capex", "1577", "million"]),
        ("revenue of 12,019,500", ["revenue", "12019500"]),
        ("clause 7.9 and rate 3.25%", ["clause", "7.9", "rate", "3.25"]),
        ("FY2018 vs. FY2017", ["fy2018", "vs", "fy2017"]),
        ("", []),
    ],
)
def test_analyzer(text: str, tokens: list[str]) -> None:
    assert analyze(text) == tokens


def test_list_commas_are_not_thousands_separators() -> None:
    assert analyze("2021,2022") == ["2021", "2022"]
    assert analyze("1,2,3") == ["1", "2", "3"]
