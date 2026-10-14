import json
import pytest

from app.generation.grounded_answer import render_answer
from app.generation.grounding import ABSTENTION
from app.rag_pipeline import relevant_evidence
from app.retrieval.fusion import RetrievedChunk


def evidence(text="Unused goods can be returned within thirty days.", cosine=0.5):
    return RetrievedChunk("c1", "d1", "policy.md", 2, text, 0.01, "semantic_chroma",
                          {"section": "Returns", "semantic_score": cosine})


def test_synonymous_question_does_not_require_exact_sentence_match():
    assert relevant_evidence("How long do I have to send a broken item back?", [evidence(cosine=0.7)])


def test_keyword_support_does_not_need_a_high_vector_score():
    assert relevant_evidence("What is the return window?", [evidence("The return window is 30 days.", 0.01)])


def test_unrelated_question_fails_relevance_gate():
    assert not relevant_evidence("What is today's IPL score?", [evidence(cosine=0.1)])


def test_generated_paraphrase_has_exact_supporting_citation():
    item = evidence()
    raw = json.dumps({"claims": [{"text": "The return period is thirty days.", "source_id": 1, "quote": item.text}]})
    answer, citations, metrics = render_answer(raw, [item])
    assert "return period" in answer
    assert citations[0].excerpt == item.text
    assert citations[0].source_filename == "policy.md"
    assert citations[0].page_number == 2
    assert citations[0].section == "Returns"
    assert citations[0].chunk_id == "c1"
    assert metrics["citation_validity"] == 1
    assert "proxy" in metrics["method"]


@pytest.mark.parametrize("claim", [
    {"text": "Outside fact", "source_id": 99, "quote": "Outside fact"},
    {"text": "Invented fact", "source_id": 1, "quote": "Invented quote"},
    {"text": "", "source_id": 1, "quote": "Unused goods"},
])
def test_invalid_supporting_citations_fail_closed(claim):
    answer, citations, _ = render_answer(json.dumps({"claims": [claim]}), [evidence()])
    assert answer == ABSTENTION and citations == []


def test_pdf_whitespace_is_restored_to_exact_original_snippet():
    item = evidence("The return window\nis thirty days.")
    raw = json.dumps({"claims": [{"text": "Returns are allowed for thirty days.", "source_id": 1,
                                 "quote": "The return window is thirty days."}]})
    _, citations, _ = render_answer(raw, [item])
    assert citations[0].excerpt == item.text


def test_invalid_claim_does_not_discard_valid_support():
    item = evidence()
    raw = json.dumps({"claims": [
        {"text": "Returns allowed within thirty days.", "source_id": 1, "quote": item.text},
        {"text": "Invented", "source_id": 99, "quote": "Invented"},
    ]})
    answer, citations, metrics = render_answer(raw, [item])
    assert answer != ABSTENTION and len(citations) == 1
    assert "Invented" not in answer and metrics["rejected_claims"] == 1


def test_overview_can_complete_a_short_selection_with_exact_source_quotes():
    items = [evidence("Support is open Monday through Friday."),
             evidence("Repairs require a service ticket."), evidence("Standard delivery takes six days.")]
    for i, item in enumerate(items):
        item.chunk_id = str(i)
    answer, citations, metrics = render_answer('{"claims": []}', items, overview=True)
    assert len(citations) == 3 and len(answer.splitlines()) == 3
    assert metrics["summary_source_quote_fallback_points"] == 3


def test_backend_selects_snippet_for_generated_claim_without_copying_quote():
    item = evidence("Delivery takes six days. The warranty lasts 31 months.")
    raw = json.dumps({"claims": [{"text": "The warranty period is 31 months.", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, [item])
    assert "31 months" in answer
    assert citations[0].excerpt == "The warranty lasts 31 months."


def test_invented_numeric_value_is_not_rendered():
    raw = json.dumps({"claims": [{"text": "The warranty lasts 99 months.", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, [evidence("The warranty lasts 31 months.")])
    assert answer == ABSTENTION and not citations


def test_citation_section_follows_the_quote_not_an_adjacent_expanded_section():
    item = evidence("Delivery takes six days. The warranty lasts 31 months.")
    item.metadata["source_units"] = [
        {"start": 0, "end": 23, "section": "Delivery"},
        {"start": 24, "end": len(item.text), "section": "Warranty"},
    ]
    raw = json.dumps({"claims": [{"text": "The warranty lasts 31 months.", "source_id": 1}]})
    _, citations, _ = render_answer(raw, [item])
    assert citations[0].section == "Warranty"


def test_repeated_brand_alone_is_not_support_for_a_missing_fact():
    items = [evidence(f"The Solara equipment supports routine maintenance procedure {i}.", 0.25) for i in range(4)]
    assert not relevant_evidence("Who is the owner of Solara?", items)


def test_numeric_snippet_selection_prefers_the_actual_value_over_a_tied_topic_sentence():
    item = evidence("The API endpoint /v2/search accepts a limit. The default limit is 25.")
    raw = json.dumps({"claims": [{"text": "The default API search limit is 25.", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, [item])
    assert "25" in answer and "25" in citations[0].excerpt


def test_multi_fact_claim_can_cite_the_exact_span_containing_both_values():
    item = evidence("Delivery takes six days. The warranty lasts 31 months. The exchange window is 45 days.")
    raw = json.dumps({"claims": [{"text": "Warranty lasts 31 months and exchanges are allowed for 45 days.", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, [item])
    assert answer != ABSTENTION
    assert citations[0].excerpt == "The warranty lasts 31 months. The exchange window is 45 days."


def test_wrong_in_range_model_source_is_aligned_to_actual_retrieved_support():
    unrelated = evidence("Invoices must be paid within fourteen days.")
    unrelated.chunk_id = "invoice"
    related = evidence("The warranty lasts 31 months.")
    related.chunk_id = "warranty"
    raw = json.dumps({"claims": [{"text": "The warranty lasts 31 months.", "source_id": 1}]})
    answer, citations, metrics = render_answer(raw, [unrelated, related])
    assert answer != ABSTENTION
    assert citations[0].chunk_id == "warranty"
    assert metrics["source_id_corrections"] == 1


def test_generated_template_without_source_support_is_not_displayed():
    raw = json.dumps({"claims": [{"text": "(your concise answer)", "source_id": 1}]})
    answer, citations, _ = render_answer(raw, [evidence()])
    assert answer == ABSTENTION and not citations


def test_model_cannot_turn_a_negative_source_statement_into_positive_answer():
    item = evidence("The Nimbus feature is not supported.")
    raw = json.dumps({"claims": [{"text": "The Nimbus feature is supported.", "source_id": 1}]})
    answer, citations, metrics = render_answer(raw, [item])
    assert "not supported" in answer and answer != ABSTENTION
    assert citations[0].excerpt == item.text
    assert metrics["polarity_source_quote_fallbacks"] == 1


def test_duplicate_claims_from_overlapping_chunks_are_rendered_once():
    first = evidence("The warranty lasts 31 months.")
    second = evidence(first.text)
    second.chunk_id = "overlap"
    raw = json.dumps({"claims": [
        {"text": first.text, "source_id": 1}, {"text": second.text, "source_id": 2},
    ]})
    answer, citations, _ = render_answer(raw, [first, second])
    assert len(citations) == 1 and answer.count("31 months") == 1


def test_weak_overview_generalization_is_replaced_by_supported_key_points():
    items = [evidence("Repairs require a service ticket."), evidence("The support desk is open Monday through Friday."),
             evidence("Standard delivery takes six days.")]
    for i, item in enumerate(items):
        item.chunk_id = str(i)
    raw = json.dumps({"claims": [{"text": "A service overview covers extensive operations and management.", "source_id": 1}]})
    answer, citations, metrics = render_answer(raw, items, overview=True)
    assert "extensive operations" not in answer
    assert len(citations) == 3 and metrics["faithfulness"] == 1.0


def test_evaluator_rejects_contradiction_despite_a_valid_citation():
    from scripts.evaluate import answer_polarity_matches_gold

    gold = "The Nimbus feature is not supported."
    data = {"answer": "The Nimbus feature is supported. [1](#source-test)",
            "citations": [{"excerpt": gold, "citation_id": 1, "anchor": "source-test"}]}
    assert not answer_polarity_matches_gold(data, gold)


def test_evaluator_accepts_a_short_answer_linked_to_a_longer_supporting_sentence():
    from scripts.evaluate import answer_polarity_matches_gold

    gold = "The router routes incoming messages by urgency, account tier, and language."
    data = {"answer": "By urgency, account tier, and language. [1](#source-test)",
            "citations": [{"excerpt": gold, "citation_id": 1, "anchor": "source-test"}]}
    assert answer_polarity_matches_gold(data, gold)
