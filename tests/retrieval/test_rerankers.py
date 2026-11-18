from collections.abc import Sequence

import pytest

from rag.domain import Chunk, Query, ScoredChunk
from rag.errors import RetrievalError
from rag.retrieval.rerankers import CrossEncoderReranker, SentenceTransformersPairScorer


def _candidate(chunk_id: str, text: str, rank: int) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="D",
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=0,
        page_end=0,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank, stage="rrf")


class WordOverlapScorer:
    """Fake cross-encoder: score = number of question words found in the passage."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]


CANDIDATES = [
    _candidate("generic", "cash flow statement overview", 1),
    _candidate("answer", "capital expenditure purchases of property plant 1577", 2),
    _candidate("partial", "capital spending discussion", 3),
]
QUERY = Query(text="capital expenditure purchases property plant")


def test_reorders_by_cross_encoder_score() -> None:
    reranked = CrossEncoderReranker(WordOverlapScorer()).rerank(QUERY, CANDIDATES, top_n=3)
    assert [c.chunk.chunk_id for c in reranked] == ["answer", "partial", "generic"]
    assert [c.rank for c in reranked] == [1, 2, 3]
    assert {c.stage for c in reranked} == {"rerank"}
    assert reranked[0].score == 5.0


def test_top_n_cuts_and_every_candidate_is_scored_once() -> None:
    scorer = WordOverlapScorer()
    reranked = CrossEncoderReranker(scorer).rerank(QUERY, CANDIDATES, top_n=1)
    assert [c.chunk.chunk_id for c in reranked] == ["answer"]
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 3
    assert scorer.calls[0][1] == (QUERY.text, CANDIDATES[1].chunk.text)


def test_ties_keep_first_stage_order() -> None:
    same = [_candidate(f"c{i}", "unrelated", i) for i in range(1, 4)]
    reranked = CrossEncoderReranker(WordOverlapScorer()).rerank(QUERY, same, top_n=3)
    assert [c.chunk.chunk_id for c in reranked] == ["c1", "c2", "c3"]


def test_empty_candidates_and_invalid_top_n() -> None:
    reranker = CrossEncoderReranker(WordOverlapScorer())
    assert reranker.rerank(QUERY, [], top_n=5) == []
    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank(QUERY, CANDIDATES, top_n=0)


def test_scorer_returning_wrong_count_is_an_error() -> None:
    class Broken:
        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [1.0]

    with pytest.raises(RetrievalError, match="1 scores for 3"):
        CrossEncoderReranker(Broken()).rerank(QUERY, CANDIDATES, top_n=3)


def test_scorer_is_lazy_and_validates_settings() -> None:
    scorer = SentenceTransformersPairScorer("org/model", revision="abc")
    assert scorer.model_id == "org/model@abc"
    assert scorer.score([]) == []  # no model load needed
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformersPairScorer("org/model", batch_size=0)


@pytest.mark.slow
def test_real_minilm_prefers_the_answering_passage() -> None:
    scorer = SentenceTransformersPairScorer(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
        device="cpu",
    )
    scores = scorer.score(
        [
            (
                "What was 3M's capital expenditure in 2018?",
                "Purchases of property, plant and equipment (PP&E) were $1,577 million in 2018.",
            ),
            ("What was 3M's capital expenditure in 2018?", "The board declared a dividend."),
        ]
    )
    assert scores[0] > scores[1]


def test_reranker_reads_the_context_header() -> None:
    scorer = WordOverlapScorer()
    contextual = CANDIDATES[0].model_copy(
        update={"chunk": CANDIDATES[0].chunk.model_copy(update={"context": "capital expenditure"})}
    )
    CrossEncoderReranker(scorer).rerank(QUERY, [contextual], top_n=1)
    assert scorer.calls[0][0][1].startswith("capital expenditure\n\n")
