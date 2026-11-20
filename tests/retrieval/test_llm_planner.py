import json
from pathlib import Path

import pytest

from rag.domain import Query
from rag.errors import GenerationError
from rag.generation.base import Completion, LLMClient, Prompt
from rag.retrieval.llm_planner import (
    PLANNER_SYSTEM,
    LLMQueryPlanner,
    PlanCache,
    plan_schema,
    prompt_fingerprint,
)

QUESTION = "What is Walmart's FY2018 DPO?"


class ScriptedLLM(LLMClient):
    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.prompts: list[Prompt] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, model_id="fake", input_tokens=1, output_tokens=1)


def _plan(*sub_queries: str) -> str:
    return json.dumps({"sub_queries": list(sub_queries)})


def test_sub_queries_become_queries_with_the_original_filters() -> None:
    llm = ScriptedLLM(_plan("Walmart FY2018 accounts payable", "Walmart FY2018 cost of sales"))
    planner = LLMQueryPlanner(llm)
    out = planner.rewrite(Query(text=QUESTION, filters={"company": "Walmart"}))

    assert [q.text for q in out] == [
        "Walmart FY2018 accounts payable",
        "Walmart FY2018 cost of sales",
    ]
    assert all(q.filters == {"company": "Walmart"} for q in out)
    (prompt,) = llm.prompts
    assert prompt.system == PLANNER_SYSTEM
    assert prompt.user == f"Question: {QUESTION}"
    assert prompt.response_schema == plan_schema(4, 200)
    assert prompt.response_schema["properties"]["sub_queries"]["items"]["maxLength"] == 200
    assert (planner.calls, planner.failures) == (1, 0)


def test_output_is_cleaned_deduplicated_capped_and_never_repeats_the_question() -> None:
    llm = ScriptedLLM(
        _plan("  Walmart   AP  ", "walmart ap", QUESTION, "", "x" * 50, "third", "fourth")
    )
    out = LLMQueryPlanner(llm, max_sub_queries=3, max_chars=10).rewrite(Query(text=QUESTION))
    assert [q.text for q in out] == ["Walmart AP", "x" * 10, "third"]


def test_prompt_never_asks_for_numbers_or_answers() -> None:
    assert "Never include numbers" in PLANNER_SYSTEM
    assert "Do not answer the question" in PLANNER_SYSTEM


@pytest.mark.parametrize(
    "reply",
    [
        GenerationError("ollama down"),
        "not json",
        json.dumps({"other": []}),
        json.dumps({"sub_queries": "one string"}),
        json.dumps(["a list"]),
    ],
)
def test_failures_fall_back_to_no_sub_queries_and_are_counted(reply: str | Exception) -> None:
    planner = LLMQueryPlanner(ScriptedLLM(reply))
    assert planner.rewrite(Query(text=QUESTION)) == []
    assert planner.failures == 1


def test_cache_avoids_repeat_calls_survives_reload_and_skips_failures(tmp_path: Path) -> None:
    path = tmp_path / "plans.json"
    llm = ScriptedLLM(_plan("Walmart AP"), GenerationError("down"))
    planner = LLMQueryPlanner(llm, cache=PlanCache(path))

    first = planner.rewrite(Query(text=QUESTION))
    again = planner.rewrite(Query(text=QUESTION))
    assert [q.text for q in first] == [q.text for q in again] == ["Walmart AP"]
    assert (planner.calls, planner.cache_hits) == (1, 1)

    assert planner.rewrite(Query(text="Other question?")) == []  # failure: not cached
    reloaded = PlanCache(path)
    assert reloaded.get(QUESTION) == ["Walmart AP"]
    assert reloaded.get("Other question?") is None


def test_cache_file_is_named_by_model_and_prompt(tmp_path: Path) -> None:
    schema = plan_schema(4, 200)
    cache = PlanCache.for_model(tmp_path, "llama3.1:8b", schema)
    assert cache.path == tmp_path / f"llama3-1-8b-{prompt_fingerprint(PLANNER_SYSTEM, schema)}.json"
    assert prompt_fingerprint("other prompt", schema) != prompt_fingerprint(PLANNER_SYSTEM, schema)
    # a schema change (e.g. a new length limit) must not reuse old plans
    assert prompt_fingerprint(PLANNER_SYSTEM, plan_schema(4, 100)) != prompt_fingerprint(
        PLANNER_SYSTEM, schema
    )


@pytest.mark.parametrize(("field", "value"), [("max_sub_queries", 0), ("max_chars", 0)])
def test_rejects_bad_limits(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        LLMQueryPlanner(ScriptedLLM(), **{field: value})  # type: ignore[arg-type]
