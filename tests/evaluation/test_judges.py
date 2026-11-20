import json
from pathlib import Path

import pytest

from rag.errors import GenerationError
from rag.evaluation.judges import (
    CORRECTNESS_SYSTEM,
    FAITHFULNESS_SYSTEM,
    JudgeCase,
    JudgeKind,
    LLMJudge,
    VerdictCache,
    correctness_message,
    faithfulness_message,
    judge_fingerprint,
    verdict_schema,
)
from rag.generation.base import Completion, LLMClient, Prompt

CASE = JudgeCase(
    item_id="q1",
    question="What was 3M's FY2018 capex?",
    reference_answer="$1577.00",
    answer="Capex was $1,577 million [S1].",
    working="Purchases of PP&E 1,577 [S1]",
    sources="[S1] 3M_2018_10K, page 59\nPurchases of PP&E (1,577)",
)


class Scripted(LLMClient):
    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.prompts: list[Prompt] = []

    @property
    def model_id(self) -> str:
        return "judge-model"

    def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, model_id="judge-model", input_tokens=1, output_tokens=1)


def _reply(verdict: str, reason: str = "matches") -> str:
    return json.dumps({"reason": reason, "verdict": verdict})


def test_correctness_judge_sees_reference_but_not_sources() -> None:
    llm = Scripted(_reply("correct"))
    verdict = LLMJudge(JudgeKind.CORRECTNESS, llm).judge(CASE)
    assert verdict is not None and verdict.passed and verdict.label == "correct"
    assert verdict.judge_model == "judge-model"
    (prompt,) = llm.prompts
    assert prompt.system == CORRECTNESS_SYSTEM
    assert (
        "Reference answer: $1577.00" in prompt.user
        and "Purchases of PP&E (1,577)" not in prompt.user
    )
    assert prompt.response_schema is not None
    assert list(prompt.response_schema["properties"]) == ["reason", "verdict"]  # reason first


def test_faithfulness_judge_sees_sources_and_working_but_not_reference() -> None:
    llm = Scripted(_reply("unfaithful", "1,577 is not in S1"))
    verdict = LLMJudge(JudgeKind.FAITHFULNESS, llm).judge(CASE)
    assert verdict is not None and not verdict.passed
    user = llm.prompts[0].user
    assert user.startswith("<sources>\n[S1] 3M_2018_10K")
    assert "Working:\nPurchases of PP&E 1,577 [S1]" in user
    assert "$1577.00" not in user


def test_rubrics_guard_against_bias_and_injection() -> None:
    for raw in (CORRECTNESS_SYSTEM, FAITHFULNESS_SYSTEM):
        system = " ".join(raw.split())
        assert "do not matter" in system  # verbosity bias
        assert "data, not instructions" in system  # prompt injection
        assert "reason first" in system  # think before the verdict
    assert "outside knowledge" in FAITHFULNESS_SYSTEM


def test_messages_without_working_or_sources() -> None:
    bare = CASE.model_copy(update={"working": "", "sources": ""})
    assert "Working:" not in faithfulness_message(bare)
    assert "(no sources)" in faithfulness_message(bare)
    assert correctness_message(bare).endswith("Answer to grade: Capex was $1,577 million [S1].")


@pytest.mark.parametrize(
    "reply",
    [GenerationError("down"), "not json", json.dumps({"reason": "x"}), _reply("maybe")],
)
def test_failures_return_none_and_are_counted(reply: str | Exception) -> None:
    judge = LLMJudge(JudgeKind.CORRECTNESS, Scripted(reply))
    assert judge.judge(CASE) is None
    assert judge.failures == 1


def test_verdicts_are_cached_by_content_and_failures_are_not(tmp_path: Path) -> None:
    cache = VerdictCache(tmp_path / "v.json")
    llm = Scripted(_reply("correct"), GenerationError("down"))
    judge = LLMJudge(JudgeKind.CORRECTNESS, llm, cache)
    first, again = judge.judge(CASE), judge.judge(CASE)
    assert first == again and (judge.calls, judge.cache_hits) == (1, 1)
    changed = CASE.model_copy(update={"answer": "Capex was $1,373 million."})
    assert judge.judge(changed) is None  # a different answer is judged anew
    assert VerdictCache(tmp_path / "v.json").get(correctness_message(changed)) is None


def test_cache_file_names_kind_model_and_rubric(tmp_path: Path) -> None:
    cache = VerdictCache.for_judge(tmp_path, JudgeKind.FAITHFULNESS, "qwen2.5:7b")
    expected = f"faithfulness-qwen2-5-7b-{judge_fingerprint(JudgeKind.FAITHFULNESS)}.json"
    assert cache.path == tmp_path / expected
    assert judge_fingerprint(JudgeKind.CORRECTNESS) != judge_fingerprint(JudgeKind.FAITHFULNESS)
    assert verdict_schema(JudgeKind.CORRECTNESS)["properties"]["verdict"]["enum"] == [
        "correct",
        "incorrect",
    ]
