import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import rag_pipeline
from app.generation.grounding import ABSTENTION
from app.generation.synthesis import answer_plan
from app.retrieval.fusion import RetrievedChunk


def chunk(text, number=1):
    return RetrievedChunk(f"c{number}", "d1", "policy.txt", None, text, 0.5, "semantic_chroma", {"semantic_score": 0.7})


def run(monkeypatch, question, chunks, outputs):
    retrieve = Mock(return_value=chunks)
    monkeypatch.setattr(rag_pipeline, "RetrievalService", lambda: SimpleNamespace(retrieve=retrieve, search_diagnostics={}))
    generator = Mock(side_effect=[(json.dumps({"claims": claims}), "test-model", 5) for claims in outputs])
    result = rag_pipeline.RAGPipeline(generator).run(None, "user", question, ["d1"], "test-model", "quick")
    assert retrieve.call_count == 1
    return result, generator


def claim(text, number=1):
    return {"text": text, "source_id": number}


def test_simple_fact_is_clear_and_cited(monkeypatch):
    text = "The warranty lasts 31 months."
    result, generator = run(monkeypatch, "What is the warranty period?", [chunk(text)], [[claim(text)]])
    assert text in result.answer and result.citations
    assert generator.call_count == 1


@pytest.mark.parametrize("separate", [False, True])
def test_comparison_covers_same_or_separate_chunks(monkeypatch, separate):
    first, second = "Alpha shipping takes 5 days.", "Beta shipping takes 9 days."
    chunks = [chunk(first), chunk(second, 2)] if separate else [chunk(first + " " + second)]
    result, _ = run(monkeypatch, "Compare Alpha shipping and Beta shipping.", chunks,
                    [[claim(first), claim(second, 2 if separate else 1)]])
    assert "5 days" in result.answer and "9 days" in result.answer
    assert result.answerability == "answerable"
    assert len(result.metadata["completeness"]["covered_parts"]) == 2


def test_multipart_retry_covers_omitted_supported_part(monkeypatch):
    first, second = "The warranty lasts 31 months.", "Delivery takes 6 days."
    result, generator = run(monkeypatch, "Explain warranty period and delivery time.", [chunk(first), chunk(second, 2)],
                            [[claim(first)], [claim(first), claim(second, 2)]])
    assert "31 months" in result.answer and "6 days" in result.answer
    assert result.metadata["retry_count"] == 1
    assert result.metadata["model_latency_ms"] == 10
    assert generator.call_args.args[0] == "Explain warranty period and delivery time."
    assert "delivery time" in generator.call_args.args[4]
    assert generator.call_args_list[0].args[1] is generator.call_args_list[1].args[1]


def test_only_one_supported_part_returns_partial_and_missing_note(monkeypatch):
    text = "The warranty lasts 31 months."
    result, generator = run(monkeypatch, "Explain warranty period and delivery time.", [chunk(text)], [[claim(text)]])
    assert "31 months" in result.answer
    assert "delivery time: not found in the uploaded document context" in result.answer
    assert result.answerability == "partial" and generator.call_count == 1


def test_retry_limit_preserves_valid_first_answer_without_false_absence_claim(monkeypatch):
    first, second = "The warranty lasts 31 months.", "Delivery takes 6 days."
    result, generator = run(monkeypatch, "Explain warranty period and delivery time.", [chunk(first), chunk(second, 2)],
                            [[claim(first)], []])
    assert generator.call_count == 2
    assert "31 months" in result.answer and result.citations
    assert "supporting passages were retrieved" in result.answer
    assert "delivery time: not found" not in result.answer
    assert result.answerability == "partial"


def test_short_factual_answer_gets_one_synthesis_retry(monkeypatch):
    text = "The warranty lasts 31 months."
    result, generator = run(monkeypatch, "What is the warranty period?", [chunk(text)],
                            [[claim("31 months.")], [claim(text)]])
    assert generator.call_count == 2 and text in result.answer


def test_unrelated_question_never_calls_generator(monkeypatch):
    item = chunk("The warranty lasts 31 months.")
    item.metadata["semantic_score"] = 0.1
    result, generator = run(monkeypatch, "What is today's IPL score?", [item], [])
    assert result.answer == ABSTENTION and not result.citations
    assert generator.call_count == 0


def test_summary_returns_three_to_five_grounded_bullets(monkeypatch):
    texts = ["Support is open Monday through Friday.", "Repairs require a service ticket.", "Delivery takes six days."]
    claims = [claim(text, i + 1) for i, text in enumerate(texts)]
    result, _ = run(monkeypatch, "Summarize the uploaded document.", [chunk(t, i + 1) for i, t in enumerate(texts)], [claims])
    assert 3 <= len(result.answer.splitlines()) <= 5
    assert all(line.startswith("- ") for line in result.answer.splitlines())
    assert len(result.citations) == 3


def test_shared_noun_phrase_is_expanded_generically():
    assert answer_plan("Compare fresh and outstanding loan WALR in February 2026.").parts == (
        "fresh loan WALR", "outstanding loan WALR")


def test_evaluator_requires_each_expected_item():
    from scripts.evaluate import answer_content_matches
    data = {"answer": "Alpha shipping takes 5 days.", "citations": [1], "retrieval_metadata": {}}
    item = {"question": "Compare Alpha and Beta shipping.", "expected_items": [["Alpha", "5"], ["Beta", "9"]], "expected_numbers": ["5", "9"]}
    assert not answer_content_matches(data, item, "Alpha shipping takes 5 days.")
    data["answer"] += " Beta shipping takes 9 days."
    assert answer_content_matches(data, item, "Alpha shipping takes 5 days.")


def test_evaluator_does_not_require_an_incidental_endpoint_version():
    from scripts.evaluate import answer_content_matches
    data = {"answer": "The identifier field returned is ticket_id.", "citations": [1], "retrieval_metadata": {}}
    assert answer_content_matches(data, {"question": "Which field is returned when creating a ticket?"},
                                  "The POST /api/v2/tickets endpoint returns ticket_id.")


def test_multipart_evidence_split_across_chunks_is_not_rejected_by_whole_query_score(monkeypatch):
    first, second = "The warranty lasts 31 months.", "Standard delivery takes 6 days."
    chunks = [chunk(first), chunk(second, 2)]
    for item in chunks:
        item.metadata["semantic_score"] = 0.1
    result, generator = run(monkeypatch, "Explain warranty period and standard delivery time.", chunks,
                            [[claim(first), claim(second, 2)]])
    assert generator.call_count == 1 and result.answerability == "answerable"


def test_negative_absence_statement_is_not_multipart_answer_evidence(monkeypatch):
    item = chunk("The document does not discuss weather forecasts or IPL scores.")
    item.metadata["semantic_score"] = 0.1
    result, generator = run(monkeypatch, "Compare weather forecasts and IPL scores.", [item], [])
    assert result.answer == ABSTENTION and generator.call_count == 0
