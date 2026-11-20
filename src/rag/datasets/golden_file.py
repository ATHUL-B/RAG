"""Our own golden sets in ``evals/`` (CLAUDE.md §4): JSONL, one ``GoldenItem`` per line.

FinanceBench has no unanswerable questions, so refusal behaviour (2.3) is tested on sets
we write ourselves. ``evidence: []`` marks an item as unanswerable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from rag.domain import GoldenItem
from rag.errors import DatasetError


def read_golden_items(path: Path) -> list[GoldenItem]:
    if not path.exists():
        raise DatasetError(f"{path} not found")
    items: list[GoldenItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"{path}:{line_no}: invalid GoldenItem: {exc}") from exc
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{path}: duplicate item_id")
    return items
