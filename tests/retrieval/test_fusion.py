import pytest

from rag.domain import Chunk, ScoredChunk
from rag.retrieval.fusion import ReciprocalRankFusion


def _ranking(stage: str, *chunk_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=cid,
                doc_id="D",
                text=cid,
                char_start=0,
                char_end=1,
                page_start=0,
                page_end=0,
            ),
            score=1.0,
            rank=rank,
            stage=stage,
        )
        for rank, cid in enumerate(chunk_ids, start=1)
    ]


# Study guide Part 9 worked example: dense ranks P,Q,B,K,R,A; BM25 ranks B,K,A,P,R,Q.
DENSE = _ranking("dense", "P", "Q", "B", "K", "R", "A")
BM25 = _ranking("bm25", "B", "K", "A", "P", "R", "Q")


def _fused(fusion: ReciprocalRankFusion, k: int = 6) -> list[ScoredChunk]:
    return fusion.fuse([DENSE, BM25], k)


class TestGuideExample:
    def test_order_is_b_p_k_q_a_r(self) -> None:
        assert [c.chunk.chunk_id for c in _fused(ReciprocalRankFusion(k=60))] == list("BPKQAR")

    def test_scores_match_the_table(self) -> None:
        scores = {c.chunk.chunk_id: c.score for c in _fused(ReciprocalRankFusion(k=60))}
        assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)  # 0.032266
        assert scores["P"] == pytest.approx(0.032018, abs=1e-6)
        assert scores["R"] == pytest.approx(2 / 65)

    def test_small_k_widens_the_gap(self) -> None:
        scores = {c.chunk.chunk_id: c.score for c in _fused(ReciprocalRankFusion(k=1))}
        assert scores["B"] == pytest.approx(0.750)
        assert scores["P"] == pytest.approx(0.700)


def test_output_is_ranked_rrf_candidates_cut_at_k() -> None:
    fused = _fused(ReciprocalRankFusion(), k=3)
    assert [c.rank for c in fused] == [1, 2, 3]
    assert {c.stage for c in fused} == {"rrf"}


def test_chunk_in_one_list_scores_its_single_term() -> None:
    fused = ReciprocalRankFusion(k=60).fuse([_ranking("dense", "X"), _ranking("bm25", "Y", "X")], 5)
    scores = {c.chunk.chunk_id: c.score for c in fused}
    assert scores["Y"] == pytest.approx(1 / 61)
    assert scores["X"] == pytest.approx(1 / 61 + 1 / 62)


def test_weights_let_one_retriever_dominate() -> None:
    fusion = ReciprocalRankFusion(k=60, weights={"dense": 3.0, "bm25": 1.0})
    assert _fused(fusion)[0].chunk.chunk_id == "P"  # dense's favourite


def test_ties_break_by_best_rank_then_first_appearance() -> None:
    fused = ReciprocalRankFusion().fuse(
        [_ranking("dense", "A", "B"), _ranking("bm25", "B", "A")], 2
    )
    assert [c.chunk.chunk_id for c in fused] == ["A", "B"]  # equal scores, A seen first


def test_empty_inputs_and_invalid_settings() -> None:
    assert ReciprocalRankFusion().fuse([], 5) == []
    assert ReciprocalRankFusion().fuse([[], []], 5) == []
    with pytest.raises(ValueError, match="k must be >= 1"):
        ReciprocalRankFusion().fuse([DENSE], 0)
    with pytest.raises(ValueError, match="k must be >= 0"):
        ReciprocalRankFusion(k=-1)
    with pytest.raises(ValueError, match="weights"):
        ReciprocalRankFusion(weights={"dense": -1.0})
