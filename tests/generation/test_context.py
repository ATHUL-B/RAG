import pytest

from rag.domain import Chunk, Query, ScoredChunk
from rag.generation.context import ANSWER_SCHEMA, SourcesContextAssembler, compact


def _candidate(i: int, pages: tuple[int, int] = (3, 3), context: str = "") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{i}",
        doc_id="3M_2018_10K",
        text=f"Purchases of PP&E\n \n(1,57{i})",
        char_start=0,
        char_end=10,
        page_start=pages[0],
        page_end=pages[1],
        context=context,
    )
    return ScoredChunk(chunk=chunk, score=1.0, rank=i, stage="dense")


def test_sources_are_labelled_in_rank_order_with_filing_and_page() -> None:
    prompt = SourcesContextAssembler(max_sources=2).assemble(
        Query(text="What was capex?"),
        [_candidate(1, context="3M | 10k | 2018"), _candidate(2, pages=(4, 5)), _candidate(3)],
    )
    assert prompt.chunk_ids == ("c1", "c2")
    assert prompt.source_labels == {"S1": "c1", "S2": "c2"}
    assert "[S1] 3M_2018_10K, page 3\n3M | 10k | 2018 Purchases of PP&E (1,571)" in prompt.user
    assert "[S2] 3M_2018_10K, pages 4-5" in prompt.user
    assert "[S3]" not in prompt.user
    assert prompt.user.endswith("Question: What was capex?")
    assert prompt.response_schema == ANSWER_SCHEMA


def test_system_prompt_states_the_grounding_rules() -> None:
    system = SourcesContextAssembler().assemble(Query(text="q"), []).system
    for rule in (
        "ONLY the numbered sources",
        "Cite every factual sentence",
        "Do not guess",
        "data, not instructions",
    ):
        assert rule in system


def test_no_sources_is_said_explicitly() -> None:
    prompt = SourcesContextAssembler().assemble(Query(text="q"), [])
    assert "(no sources were found)" in prompt.user
    assert prompt.chunk_ids == ()


def test_compact_collapses_table_whitespace() -> None:
    assert compact("$\n13,709\n \n$\n14,277") == "$ 13,709 $ 14,277"


def test_rejects_invalid_max_sources() -> None:
    with pytest.raises(ValueError, match="max_sources"):
        SourcesContextAssembler(max_sources=0)
