from app.retrieval.fusion import RetrievedChunk, reciprocal_rank_fusion
from app.retrieval.keyword_search import bm25_scores


def item(chunk_id: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "doc", "a.md", None, "text", score, source)


def test_bm25_finds_exact_terms():
    scores = bm25_scores("KD-401", ["nothing here", "error KD-401 upload failed"])
    assert scores[1] > scores[0]


def test_rrf_deduplicates_chunks():
    fused = reciprocal_rank_fusion([[item("a", 1, "semantic")], [item("a", 2, "keyword"), item("b", 1, "keyword")]])
    assert [row.chunk_id for row in fused].count("a") == 1
    assert fused[0].chunk_id == "a"

