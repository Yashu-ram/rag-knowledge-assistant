from pathlib import Path

import pytest

from app.generation.grounding import supporting_passages, validate_selection
from app.retrieval.fusion import RetrievedChunk


def evidence(text=None):
    source = Path(__file__).resolve().parents[1] / "fixtures/rbi_transmission_excerpt.txt"
    return [RetrievedChunk("rbi-chunk", "rbi-doc", "rbi-policy.pdf", 48,
                           text if text is not None else source.read_text(encoding="utf-8"), 1.0, "test")]


def test_fresh_loan_walr_keeps_subject_direction_and_amount():
    question = "How did the WALR on fresh rupee loans change during H2:2025-26 up to February 2026?"
    passages = supporting_passages(question, evidence())
    assert passages
    assert all("WALR" in p.quote and "increased by 5 bps" in p.quote for p in passages)
    assert all("WADTDR" not in p.quote for p in passages)
    answer, citations, _ = validate_selection('{"passage_ids": [1]}', passages)
    assert "fresh rupee loans increased by 5 bps" in answer
    assert citations[0].page_number == 48
    assert citations[0].excerpt in evidence()[0].text


@pytest.mark.parametrize("question,expected", [
    ("How much was the cumulative policy rate reduction in the easing cycle from February 2025?", "125 basis points"),
    ("By how much did the median 1-year MCLR of scheduled commercial banks decline in H2:2025-26?", "declined by 20 bps"),
])
def test_other_rate_questions_keep_the_correct_subject(question, expected):
    passages = supporting_passages(question, evidence())
    assert passages
    assert all(expected in p.quote for p in passages)


def test_date_overlap_cannot_support_wrong_subject_or_trailing_fragment():
    text = "20 bps in H2:2025-26 (up to February 2026), while"
    assert not supporting_passages("How did the WALR on fresh rupee loans change during H2:2025-26 up to February 2026?", evidence(text))


def test_full_cycle_change_cannot_answer_half_year_question():
    text = "In the current easing cycle (up to February 2026), the WALR for fresh rupee loans declined by 89 bps. H2:2025-26 data are discussed elsewhere."
    assert not supporting_passages("How did WALR on fresh rupee loans change in H2:2025-26?", evidence(text))


def test_february_observation_cannot_answer_january_question():
    assert not supporting_passages(
        "How did WALR on fresh rupee loans change in H2:2025-26 up to January 2026?", evidence())
