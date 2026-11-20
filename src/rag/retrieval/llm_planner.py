"""LLM query planner: decompose a question into focused sub-queries (Part 7; HLD 012).

One embedding cannot sit near several evidence pages at once. A DPO question needs
accounts payable from the balance sheet *and* cost of sales from the income statement, so
the single question vector lands between them. The planner asks a small local model for
one short search per fact the question needs. ``MultiQueryRetriever`` searches each one,
alongside the original and the rule-cleaned question, and fuses the lists with RRF.

Safety rules from Part 7:

- The original question is always searched too (the multi-query retriever keeps it).
- Filters stay rule-based and come from the original question: a wrong LLM-guessed filter
  hides the answer completely. The planner only adds search texts.
- The prompt asks for search phrases built from the question's own terms and standard
  line-item names, never numbers or facts. Output is schema-constrained, deduplicated,
  capped in number and length.
- Any failure (timeout, error, malformed JSON) means "no sub-queries": retrieval falls
  back to the rules, the failure is counted, and it is never cached.
- Plans are cached per (model, prompt): re-runs are free, fast and identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Final

from rag.domain import Query
from rag.errors import GenerationError
from rag.generation.base import LLMClient, Prompt
from rag.retrieval.base import QueryRewriter
from rag.storage import atomic_write_bytes

logger = logging.getLogger(__name__)

PLANNER_SYSTEM: Final = """\
You plan searches over company SEC filings (10-K, 10-Q, 8-K, earnings releases).
Given a question, write the search queries that would find every page needed to answer it.

Rules:
- One query per distinct fact, line item or financial statement the answer needs.
  Example: a ratio of balance-sheet and income-statement items needs one query for each item.
- Each query is a short phrase (under 15 words) naming the company, the fiscal year(s) and
  the line item or topic, using standard financial statement wording
  (e.g. "Walmart FY2018 accounts payable balance sheet").
- Use only the company, years and concepts in the question. Never include numbers, values
  or facts that are not in the question. Do not answer the question.
- If the question needs only one fact, return one query.
"""

_NON_SLUG: Final = re.compile(r"[^a-z0-9]+")


def plan_schema(max_sub_queries: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max_sub_queries,
            }
        },
        "required": ["sub_queries"],
        "additionalProperties": False,
    }


def prompt_fingerprint(system: str = PLANNER_SYSTEM) -> str:
    """Changes whenever the instructions change, so cached plans never outlive their prompt."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]


class PlanCache:
    """Question text → sub-queries, persisted as one JSON file per (model, prompt)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._plans: dict[str, list[str]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    @classmethod
    def for_model(cls, cache_dir: Path, model_id: str) -> PlanCache:
        slug = _NON_SLUG.sub("-", model_id.lower()).strip("-")
        return cls(cache_dir / f"{slug}-{prompt_fingerprint()}.json")

    @property
    def path(self) -> Path:
        return self._path

    def get(self, question: str) -> list[str] | None:
        return self._plans.get(question)

    def put(self, question: str, sub_queries: list[str]) -> None:
        self._plans[question] = sub_queries
        body = json.dumps(self._plans, indent=1, sort_keys=True, ensure_ascii=False)
        atomic_write_bytes(self._path, body.encode("utf-8"))


class LLMQueryPlanner(QueryRewriter):
    def __init__(
        self,
        llm: LLMClient,
        max_sub_queries: int = 4,
        max_chars: int = 200,
        cache: PlanCache | None = None,
    ) -> None:
        if max_sub_queries < 1:
            raise ValueError("max_sub_queries must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self._llm = llm
        self._max_sub_queries = max_sub_queries
        self._max_chars = max_chars
        self._cache = cache
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    def rewrite(self, query: Query) -> list[Query]:
        sub_queries = self._plan(query.text)
        return [query.model_copy(update={"text": text}) for text in sub_queries]

    def _plan(self, question: str) -> list[str]:
        if self._cache is not None and (cached := self._cache.get(question)) is not None:
            self.cache_hits += 1
            return cached
        self.calls += 1
        prompt = Prompt(
            system=PLANNER_SYSTEM,
            user=f"Question: {question}",
            chunk_ids=(),
            response_schema=plan_schema(self._max_sub_queries),
        )
        try:
            completion = self._llm.complete(prompt)
            sub_queries = self._validated(json.loads(completion.text), question)
        except (GenerationError, json.JSONDecodeError, TypeError, KeyError) as exc:
            self.failures += 1
            logger.warning("query planner failed; using rules only", extra={"error": str(exc)})
            return []
        if self._cache is not None:
            self._cache.put(question, sub_queries)
        return sub_queries

    def _validated(self, data: Any, question: str) -> list[str]:
        raw = data["sub_queries"]
        if not isinstance(raw, list):
            raise TypeError("sub_queries is not a list")
        original = " ".join(question.split()).lower()
        seen: set[str] = set()
        kept: list[str] = []
        for item in raw:
            full = " ".join(str(item).split())
            if full.lower() == original:  # compare before truncating: a cut copy is still a copy
                continue
            text = full[: self._max_chars].strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                kept.append(text)
        return kept[: self._max_sub_queries]
