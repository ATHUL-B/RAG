from pathlib import Path

import pytest

from rag.domain import GoldenItem, SourceSpan
from rag.errors import DatasetError
from rag.evaluation.splits import (
    SplitName,
    default_stratum,
    load_split,
    make_split,
    save_split,
    select,
)


def _item(i: int, company: str, kind: str = "a", evidence: str = "single") -> GoldenItem:
    return GoldenItem(
        item_id=f"q{i:03d}",
        question="?",
        reference_answer="x",
        evidence=(SourceSpan(doc_id="D", page_index=0),),
        tags=frozenset({f"type:{kind}", f"evidence:{evidence}"}),
        metadata={"company": company},
    )


def _corpus() -> list[GoldenItem]:
    """12 companies x 3 questions; two question types alternate across companies."""
    return [
        _item(
            c * 3 + j,
            f"co{c}",
            kind="a" if c % 2 else "b",
            evidence="multi" if j == 0 else "single",
        )
        for c in range(12)
        for j in range(3)
    ]


def test_split_is_disjoint_complete_and_grouped_by_company() -> None:
    items = _corpus()
    split = make_split(items, test_fraction=1 / 3)
    assert set(split.dev).isdisjoint(split.test)
    assert set(split.dev) | set(split.test) == {i.item_id for i in items}
    company = {i.item_id: i.metadata["company"] for i in items}
    assert {company[i] for i in split.dev}.isdisjoint({company[i] for i in split.test})


def test_split_balances_strata_toward_the_test_fraction() -> None:
    split = make_split(_corpus(), test_fraction=1 / 3)
    # each of 4 strata has 6 or 12 items; test should hold about a third of each
    assert split.strata[SplitName.TEST] == {
        "evidence:multi|type:a": 2,
        "evidence:multi|type:b": 2,
        "evidence:single|type:a": 4,
        "evidence:single|type:b": 4,
    }
    assert len(split.test) == 12


def test_split_is_deterministic_and_seed_dependent() -> None:
    items = _corpus()
    assert make_split(items, seed=0) == make_split(items, seed=0)
    assert make_split(list(reversed(items)), seed=0).test == make_split(items, seed=0).test
    assert {make_split(items, seed=s).test for s in range(5)} != {make_split(items).test}


def test_select_keeps_original_order_and_all_returns_everything(tmp_path: Path) -> None:
    items = _corpus()
    split = make_split(items)
    path = tmp_path / "split.json"
    save_split(split, path)
    loaded = load_split(path)
    assert loaded == split
    dev = select(items, loaded, SplitName.DEV)
    assert [i.item_id for i in dev] == [i.item_id for i in items if i.item_id in set(split.dev)]
    assert select(items, loaded, SplitName.ALL) == items


def test_select_rejects_ids_missing_from_the_dataset() -> None:
    items = _corpus()
    split = make_split(items)
    with pytest.raises(DatasetError, match="not in the dataset"):
        select(items[1:], split, SplitName.DEV if items[0].item_id in split.dev else SplitName.TEST)


def test_missing_split_file_explains_how_to_create_it(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="make_split"):
        load_split(tmp_path / "nope.json")


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_rejects_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        make_split(_corpus(), test_fraction=fraction)


def test_rejects_duplicates_empty_and_missing_company() -> None:
    item = _item(1, "co")
    with pytest.raises(ValueError, match="unique"):
        make_split([item, item])
    with pytest.raises(ValueError, match="no items"):
        make_split([])
    with pytest.raises(DatasetError, match="no company"):
        make_split([item.model_copy(update={"metadata": {}})])


def test_default_stratum_uses_type_and_evidence_tags_only() -> None:
    item = _item(1, "co").model_copy(
        update={"tags": frozenset({"type:a", "evidence:multi", "reasoning:x"})}
    )
    assert default_stratum(item) == "evidence:multi|type:a"
