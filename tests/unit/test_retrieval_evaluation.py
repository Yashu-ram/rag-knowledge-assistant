import pytest

from app.retrieval.evaluation import retrieval_scores
from app.retrieval.keyword_search import bm25_scores, tokenize


def test_precision_and_recall_use_independent_relevance_labels():
    metrics = retrieval_scores(["a", "b", "a", "c"], {"a", "d"}, 3)
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["retrieved_total"] == 3


def test_short_result_list_does_not_inflate_precision_at_k():
    metrics = retrieval_scores(["a"], {"a"}, 6)
    assert metrics["precision_at_k"] == pytest.approx(1 / 6)
    assert metrics["retrieved_precision"] == 1.0


def test_recall_is_undefined_without_relevant_documents():
    assert retrieval_scores(["a"], set(), 6)["recall_at_k"] is None


def test_keyword_matching_handles_punctuation_and_identifiers():
    assert tokenize("KD-401, ticket_id, 4.2. PDF.") == ["kd-401", "ticket_id", "4.2", "pdf"]
    scores = bm25_scores("PDF KD-401", ["A filename is not a name.", "PDF. Error KD-401."])
    assert scores[1] > scores[0]


def test_keyword_matching_does_not_match_substrings():
    assert bm25_scores("name", ["filename field"])[0] == 0
