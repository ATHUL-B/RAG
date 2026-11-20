"""Dev/test split of the golden set (study guide, Part 13).

Every knob we tune (RRF weights, rewrite prompts, thresholds) is tuned on **dev**; **test**
is evaluated rarely, to confirm a change generalises. Tuning on the questions you report
on overfits the evaluation, the same way training on the test set overfits a model.

The split is **grouped by company**: all questions about one company land on the same
side. Questions about one company share filings, vocabulary and table layouts, so a
random split would let dev tuning "see" the test companies. Grouping makes test measure
performance on companies we never tuned on, which is what a new filing looks like in
production. Within that constraint the split balances the strata (question type x
single/multi evidence) so both halves have the same mix of difficulty.

The result is committed to ``evals/`` as a list of IDs, so it never changes silently.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import GoldenItem
from rag.errors import DatasetError
from rag.storage import atomic_write_bytes


class SplitName(StrEnum):
    DEV = "dev"
    TEST = "test"
    ALL = "all"


class Split(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    method: str
    seed: int
    dev: tuple[str, ...]
    test: tuple[str, ...]
    notes: str = ""
    strata: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Per split: stratum → count, recorded so the balance is visible in the file."""


def default_stratum(item: GoldenItem) -> str:
    """Question type x single/multi evidence, from the FinanceBench tags."""
    kinds = sorted(t for t in item.tags if t.startswith(("type:", "evidence:")))
    return "|".join(kinds)


def company_of(item: GoldenItem) -> str:
    company = item.metadata.get("company")
    if not isinstance(company, str) or not company:
        raise DatasetError(f"{item.item_id}: no company in metadata, cannot group the split")
    return company


def _rank(seed: int, key: str) -> str:
    """Deterministic pseudo-random order: the same seed gives the same split everywhere."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def make_split(
    items: Sequence[GoldenItem],
    test_fraction: float = 1 / 3,
    seed: int = 0,
    group: Callable[[GoldenItem], str] = company_of,
    stratum: Callable[[GoldenItem], str] = default_stratum,
) -> Split:
    """Assign whole groups to test greedily so test's strata approach their target counts.

    Groups are visited in seeded-hash order. A group joins test only if that brings test's
    per-stratum counts closer (squared distance) to ``test_fraction`` of each stratum's
    total; otherwise it goes to dev. Pure and deterministic.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if len({i.item_id for i in items}) != len(items):
        raise ValueError("item ids must be unique")
    if not items:
        raise ValueError("no items to split")

    groups: dict[str, list[GoldenItem]] = {}
    for item in items:
        groups.setdefault(group(item), []).append(item)
    totals = Counter(stratum(i) for i in items)
    target = {s: n * test_fraction for s, n in totals.items()}

    in_test: Counter[str] = Counter()
    test_groups: set[str] = set()
    for name in sorted(groups, key=lambda g: _rank(seed, g)):
        adds = Counter(stratum(i) for i in groups[name])
        before = sum((in_test[s] - target[s]) ** 2 for s in totals)
        after = sum((in_test[s] + adds[s] - target[s]) ** 2 for s in totals)
        if after < before:
            in_test.update(adds)
            test_groups.add(name)

    test = tuple(sorted(i.item_id for i in items if group(i) in test_groups))
    dev = tuple(sorted(i.item_id for i in items if group(i) not in test_groups))
    by_id = {i.item_id: i for i in items}
    return Split(
        method=(
            f"grouped by company, stratified by type x evidence, test_fraction={test_fraction:.3f}"
        ),
        seed=seed,
        dev=dev,
        test=test,
        strata={
            SplitName.DEV: dict(sorted(Counter(stratum(by_id[i]) for i in dev).items())),
            SplitName.TEST: dict(sorted(Counter(stratum(by_id[i]) for i in test).items())),
        },
    )


def select(items: Iterable[GoldenItem], split: Split, name: SplitName) -> list[GoldenItem]:
    """The items of one split, in their original order. ``all`` returns every item."""
    items = list(items)
    if name is SplitName.ALL:
        return items
    wanted = set(split.dev if name is SplitName.DEV else split.test)
    chosen = [i for i in items if i.item_id in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {i.item_id for i in chosen})
        raise DatasetError(f"split lists ids not in the dataset: {missing[:5]}")
    return chosen


def save_split(split: Split, path: Path) -> None:
    atomic_write_bytes(path, (json.dumps(split.model_dump(), indent=2) + "\n").encode("utf-8"))


def load_split(path: Path) -> Split:
    if not path.exists():
        raise DatasetError(f"{path} not found; run scripts/make_split.py first")
    return Split.model_validate_json(path.read_text(encoding="utf-8"))
