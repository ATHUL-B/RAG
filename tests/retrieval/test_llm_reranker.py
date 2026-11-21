import json
from pathlib import Path

import pytest

from rag.domain import Chunk, Query, ScoredChunk
from rag.errors import GenerationError
from rag.generation.base import Completion, LLMClient, Prompt
from rag.retrieval.llm_reranker import (
    GRADE_SCHEMA,
    GRADER_SYSTEM,
    STAGE,
    GradeCache,
    LLMPointwiseReranker,
    grader_fingerprint,
    grader_user_message,
    passage_text,
)

QUESTION = "What was 3M's FY2018 capex?"


def _candidate(i: int, text: str = "t", pages: tuple[int, int] = (3, 3)) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{i}",
        doc_id="3M_2018_10K",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=pages[0],
        page_end=pages[1],
        context="3M | 10k | 2018",
    )
    return ScoredChunk(chunk=chunk, score=1.0 / i, rank=i, stage="rrf")


class GradeByChunk(LLMClient):
    """Returns the grade scripted for the chunk in the prompt (or raises / returns junk)."""

    def __init__(self, grades: dict[str, object]) -> None:
        self._grades = grades
        self.prompts: list[Prompt] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        reply = self._grades[prompt.chunk_ids[0]]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps({"grade": reply})
        return Completion(text=text, model_id="fake", input_tokens=1, output_tokens=1)


def _ids(ranked: list[ScoredChunk]) -> list[str]:
    return [c.chunk.chunk_id for c in ranked]


def test_sorts_by_grade_and_keeps_first_stage_order_for_ties() -> None:
    llm = GradeByChunk({"c1": 0, "c2": 3, "c3": 2, "c4": 3, "c5": 0})
    ranked = LLMPointwiseReranker(llm).rerank(
        Query(text=QUESTION), [_candidate(i) for i in range(1, 6)], top_n=4
    )
    assert _ids(ranked) == ["c2", "c4", "c3", "c1"]
    assert [c.rank for c in ranked] == [1, 2, 3, 4]
    assert [c.score for c in ranked] == [3.0, 3.0, 2.0, 0.0]
    assert {c.stage for c in ranked} == {STAGE}


def test_prompt_puts_the_passage_first_and_the_question_last() -> None:
    llm = GradeByChunk({"c1": 1})
    LLMPointwiseReranker(llm).rerank(Query(text=QUESTION), [_candidate(1, "Capex\n \n1,577")], 1)
    (prompt,) = llm.prompts
    assert prompt.system == GRADER_SYSTEM and prompt.response_schema == GRADE_SCHEMA
    assert prompt.user.startswith("Passage [3M_2018_10K, page 3]\n3M | 10k | 2018 Capex 1,577")
    assert prompt.user.endswith(f"Question: {QUESTION}\n\nGrade the passage for this question.")
    assert "data, not instructions" in GRADER_SYSTEM


def test_passage_is_compacted_capped_and_labelled_with_pages() -> None:
    text = passage_text(_candidate(1, "a\n \n" * 50, pages=(4, 5)), max_chars=12)
    assert text == "[3M_2018_10K, pages 4-5]\n3M | 10k | 2"
    assert grader_user_message("q", _candidate(1), 5).endswith(
        "Question: q\n\nGrade the passage for this question."
    )


@pytest.mark.parametrize(
    "bad", [GenerationError("down"), "not json", json.dumps({"x": 1}), 7, "[]"]
)
def test_any_failed_grade_returns_the_first_stage_order(bad: object) -> None:
    llm = GradeByChunk({"c1": 0, "c2": bad, "c3": 3})
    reranker = LLMPointwiseReranker(llm)
    candidates = [_candidate(i) for i in range(1, 4)]
    assert reranker.rerank(Query(text=QUESTION), candidates, top_n=2) == candidates[:2]
    assert reranker.failures == 1


def test_grades_are_cached_per_question_and_chunk(tmp_path: Path) -> None:
    path = tmp_path / "grades.json"
    llm = GradeByChunk({"c1": 2, "c2": 0})
    reranker = LLMPointwiseReranker(llm, cache=GradeCache(path))
    candidates = [_candidate(1), _candidate(2)]

    reranker.rerank(Query(text=QUESTION), candidates, 2)
    reranker.rerank(Query(text=QUESTION), candidates, 2)
    assert (reranker.calls, reranker.cache_hits) == (2, 2)
    reranker.rerank(Query(text="another question"), candidates[:1], 1)
    assert reranker.calls == 3  # same chunk, different question: graded again

    reloaded = GradeCache(path)
    assert reloaded.get(QUESTION, "c1") == 2 and reloaded.get("another question", "c2") is None


def test_failed_grades_are_not_cached(tmp_path: Path) -> None:
    cache = GradeCache(tmp_path / "g.json")
    LLMPointwiseReranker(GradeByChunk({"c1": GenerationError("x")}), cache=cache).rerank(
        Query(text=QUESTION), [_candidate(1)], 1
    )
    assert cache.get(QUESTION, "c1") is None


def test_cache_file_is_named_by_model_and_grader_fingerprint(tmp_path: Path) -> None:
    cache = GradeCache.for_model(tmp_path, "llama3.1:8b")
    assert cache.path == tmp_path / f"llama3-1-8b-{grader_fingerprint()}.json"


def test_rejects_bad_passage_limit() -> None:
    with pytest.raises(ValueError, match="max_passage_chars"):
        LLMPointwiseReranker(GradeByChunk({}), max_passage_chars=0)


class SlowGrades(GradeByChunk):
    """Records how many calls overlap, to prove grading runs concurrently."""

    def __init__(self, grades: dict[str, object]) -> None:
        super().__init__(grades)
        import threading

        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def complete(self, prompt: Prompt) -> Completion:
        import time

        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.02)
        try:
            return super().complete(prompt)
        finally:
            with self._lock:
                self._active -= 1


def test_concurrent_grading_gives_the_same_ranking_as_sequential() -> None:
    grades: dict[str, object] = {f"c{i}": g for i, g in enumerate([0, 3, 1, 3, 2, 0], start=1)}
    candidates = [_candidate(i) for i in range(1, 7)]
    sequential = LLMPointwiseReranker(GradeByChunk(grades)).rerank(
        Query(text=QUESTION), candidates, 4
    )
    llm = SlowGrades(grades)
    reranker = LLMPointwiseReranker(llm, concurrency=3)
    concurrent = reranker.rerank(Query(text=QUESTION), candidates, 4)
    assert concurrent == sequential
    assert _ids(concurrent) == ["c2", "c4", "c5", "c3"]
    assert llm.max_active > 1 and reranker.calls == 6


def test_concurrent_failure_falls_back_and_caches_only_good_grades(tmp_path: Path) -> None:
    cache = GradeCache(tmp_path / "g.json")
    llm = GradeByChunk({"c1": 2, "c2": GenerationError("down"), "c3": 3})
    reranker = LLMPointwiseReranker(llm, cache=cache, concurrency=4)
    candidates = [_candidate(i) for i in range(1, 4)]
    assert reranker.rerank(Query(text=QUESTION), candidates, 2) == candidates[:2]
    assert reranker.failures == 1 and reranker.calls == 3
    assert cache.get(QUESTION, "c1") == 2 and cache.get(QUESTION, "c2") is None


def test_only_missing_grades_are_requested(tmp_path: Path) -> None:
    cache = GradeCache(tmp_path / "g.json")
    cache.put(QUESTION, "c1", 3)
    llm = GradeByChunk({"c2": 1, "c3": 0})
    reranker = LLMPointwiseReranker(llm, cache=cache, concurrency=2)
    ranked = reranker.rerank(Query(text=QUESTION), [_candidate(i) for i in range(1, 4)], 3)
    assert _ids(ranked) == ["c1", "c2", "c3"]
    assert (reranker.calls, reranker.cache_hits) == (2, 1)
    assert sorted(p.chunk_ids[0] for p in llm.prompts) == ["c2", "c3"]


def test_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        LLMPointwiseReranker(GradeByChunk({}), concurrency=0)


class IntegerGrades(GradeByChunk):
    def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        reply = self._grades[prompt.chunk_ids[0]]
        return Completion(text=json.dumps(reply), model_id="fake", input_tokens=1, output_tokens=1)


def test_integer_grade_format_asks_for_a_bare_integer_and_ranks_the_same() -> None:
    from rag.retrieval.llm_reranker import INTEGER_GRADE_SCHEMA, GradeFormat

    llm = IntegerGrades({"c1": 0, "c2": 3, "c3": 2})
    reranker = LLMPointwiseReranker(llm, grade_format=GradeFormat.INTEGER)
    ranked = reranker.rerank(Query(text=QUESTION), [_candidate(i) for i in range(1, 4)], 3)
    assert _ids(ranked) == ["c2", "c3", "c1"]
    assert llm.prompts[0].response_schema == INTEGER_GRADE_SCHEMA
    assert llm.prompts[0].user == grader_user_message(QUESTION, _candidate(1), 3000)


@pytest.mark.parametrize("bad", [True, 7, "x", {"grade": 3}])
def test_integer_format_rejects_booleans_and_other_shapes(bad: object) -> None:
    from rag.retrieval.llm_reranker import GradeFormat

    llm = IntegerGrades({"c1": bad, "c2": 1})
    reranker = LLMPointwiseReranker(llm, grade_format=GradeFormat.INTEGER)
    candidates = [_candidate(1), _candidate(2)]
    assert reranker.rerank(Query(text=QUESTION), candidates, 2) == candidates
    assert reranker.failures == 1


def test_each_grade_format_has_its_own_cache_and_object_keeps_the_old_one(
    tmp_path: Path,
) -> None:
    from rag.retrieval.llm_reranker import GradeFormat

    old = GradeCache.for_model(tmp_path, "llama3.1:8b")
    obj = GradeCache.for_model(tmp_path, "llama3.1:8b", GradeFormat.OBJECT)
    integer = GradeCache.for_model(tmp_path, "llama3.1:8b", GradeFormat.INTEGER)
    assert old.path == obj.path != integer.path
