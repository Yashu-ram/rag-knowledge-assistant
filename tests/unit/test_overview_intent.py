import pytest

from app.generation.overview import is_overview_question


@pytest.mark.parametrize("question", [
    "What is this document about?",
    "Summarize the uploaded document.",
    "Summarize the uploaded document using citations.",
    "Give me the key points.",
    "What are the main topics?",
    "Explain this document briefly.",
    "Explain briefly.",
    "Give key points.",
    "Please give me a brief overview of the PDF with citations.",
])
def test_overview_intent(question):
    assert is_overview_question(question)


@pytest.mark.parametrize("question", [
    "What is today's IPL score?",
    "Summarize today's IPL score",
    "Summarize the uploaded document and give today's IPL score",
    "What is the maximum upload size?",
    "Explain the refund policy briefly",
])
def test_factual_intent_does_not_bypass_confidence(question):
    assert not is_overview_question(question)


def test_summary_prompt_requires_source_only_cited_points():
    from app.generation.model_gateway import build_prompt
    from tests.unit.test_grounding import chunk

    prompt = build_prompt("Summarize the uploaded document using citations", [
        chunk("Delivery takes five days. Returns are free for thirty days."),
    ], "quick")
    assert "based only on the retrieved context" in prompt
    assert "Do not include unsupported facts" in prompt
    assert "Cite each major point" in prompt
    assert "3-5" in prompt
    assert "chunk_id" in prompt


def test_summary_invalid_selection_still_fails_closed():
    from app.generation.grounding import ABSTENTION, summary_passages, validate_summary_selection
    from tests.unit.test_grounding import chunk

    passages = summary_passages([chunk("Delivery takes five days. Returns are free for thirty days.")])
    answer, citations, metrics = validate_summary_selection('{"passage_ids": [1, 999]}', passages)
    assert answer == ABSTENTION
    assert not citations
    assert metrics["rejected_claims"] == 1


def test_summary_ensures_chunk_diversity_even_for_long_model_selection():
    from app.generation.grounding import Passage, validate_summary_selection
    from tests.unit.test_grounding import chunk

    quotes = [f"Delivery rule {i} requires verification within {i + 1} days." for i in range(5)]
    first = chunk(" ".join(quotes))
    other = chunk("Returns are free for thirty days.", "c2")
    passages = [Passage(i + 1, first, quote) for i, quote in enumerate(quotes)]
    passages.append(Passage(6, other, other.text))
    answer, citations, metrics = validate_summary_selection('{"passage_ids": [1, 2, 3, 4, 5]}', passages)
    assert len(citations) == 5
    assert {c.chunk_id for c in citations} == {"c1", "c2"}
    assert all(line.startswith("- ") for line in answer.splitlines())
    assert metrics["faithfulness"] == 1.0
