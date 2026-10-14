import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.api.schemas import ChatRequest
from app.generation.grounded_answer import render_answer
from app.generation.grounding import ABSTENTION
from app.generation import model_gateway
from app.retrieval.fusion import RetrievedChunk


def source(text, chunk_id="policy"):
    return RetrievedChunk(chunk_id, "doc", "policy.pdf", 1, text, 0.5, "semantic_chroma", {})


@pytest.mark.parametrize("raw", [
    '{"claims": [{"text": "The cumulative policy rate reduction was 125 bps."}]}',
    '{"answer": "The cumulative policy rate reduction was 125 bps."}',
    "The cumulative policy rate reduction was 125 bps.",
    '{"claims": [{"text": "The cumulative policy rate reduction was 125 bps. [1]", "source_id": "[1]"}]}',
    '{"claims": [{"text": "The cumulative policy rate reduction was 125 bps.", "source_id": 99}]}',
])
def test_supported_answer_recovers_missing_or_malformed_source(raw):
    context = [source("The cumulative policy rate reduction was 125 basis points.")]
    answer, citations, metrics = render_answer(raw, context)
    assert "125" in answer and answer != ABSTENTION
    assert citations[0].chunk_id == "policy"
    assert citations[0].excerpt == context[0].text
    assert metrics["citation_validity"] == 1


def test_marker_digits_are_not_checked_as_answer_values():
    context = [source("The warranty lasts 31 months.")]
    raw = json.dumps({"claims": [{"text": "The warranty lasts 31 months. [1]", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, context)
    assert "31 months" in answer and citations


def test_recovered_claim_cannot_reverse_the_source_direction():
    answer, citations, metrics = render_answer("MCLR increased by 20 bps.",
                                              [source("MCLR declined by 20 basis points.")])
    assert "declined" in answer and "increased" not in answer
    assert citations and metrics["direction_source_quote_fallbacks"] == 1


def test_repair_chooses_support_not_first_retrieved_chunk():
    context = [source("MCLR declined by 20 basis points.", "mclr"),
               source("The cumulative policy rate reduction was 125 basis points.", "policy")]
    answer, citations, metrics = render_answer("The cumulative policy rate reduction was 125 bps.", context)
    assert citations[0].chunk_id == "policy"
    assert metrics["citations_recovered"] == 1


@pytest.mark.parametrize("raw", ["Today's IPL score is 125.",
                                  "The cumulative policy rate reduction was 999 bps.",
                                  '{"claims": []}', '{"claims": broken'])
def test_unsupported_or_empty_answer_does_not_get_automatic_citations(raw):
    answer, citations, _ = render_answer(raw, [source("The cumulative policy rate reduction was 125 basis points.")])
    assert answer == ABSTENTION and not citations


def test_raw_answer_from_failed_structured_parser_is_preserved(monkeypatch):
    llm = MagicMock()
    raw = "The cumulative policy rate reduction was 125 basis points."
    llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": None, "raw": SimpleNamespace(content=raw), "parsing_error": ValueError("source id missing")}
    monkeypatch.setattr(model_gateway, "ChatOllama", lambda **kwargs: llm)
    monkeypatch.setattr(model_gateway, "ensure_ollama_model", lambda model: model)
    result, _, _ = model_gateway.generate_answer("What was the policy rate reduction?", [], "qwen2.5:1.5b", "quick")
    assert result == raw
    assert llm.with_structured_output.call_args.kwargs["include_raw"] is True


def test_prompt_numbers_sources_and_requests_markers():
    prompt = model_gateway.build_prompt("What was the policy rate reduction?", [source("Reduction was 125 bps.")])
    assert '"source_id": 1' in prompt
    assert "[1] or [2]" in prompt


def test_single_document_request_aliases_are_not_ignored():
    body = ChatRequest(question="Test?", document_id="doc", model="qwen2.5:3b")
    assert body.document_ids == ["doc"] and body.model_name == "qwen2.5:3b"
    with pytest.raises(ValueError, match="Conflicting"):
        ChatRequest(question="Test?", document_id="one", document_ids=["two"])


def test_default_evaluation_labels_match_fixed_corpus():
    root = Path(__file__).resolve().parents[2]
    questions = json.loads((root / "tests/evaluation_questions.json").read_text(encoding="utf-8"))
    gold = json.loads((root / "tests/retrieval_gold.json").read_text(encoding="utf-8"))
    for question in questions:
        if question["expected_source"]:
            path = root / "tests/fixtures/evaluation_corpus" / question["expected_source"]
            assert gold[question["question"]] in path.read_text(encoding="utf-8")
