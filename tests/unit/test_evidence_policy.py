from app.retrieval.evidence_policy import assess_evidence
from app.retrieval.fusion import RetrievedChunk


def chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        source_filename="doc.md",
        page_number=None,
        text=text,
        score=1.0,
        source="test",
    )


def test_missing_key_term_abstains_for_unanswerable_question():
    decision = assess_evidence("Who is the CEO of KiwiDesk?", [chunk("KiwiDesk 4.2 is supported.")])
    assert decision.answerability == "unanswerable"


def test_focused_single_chunk_can_be_answerable():
    decision = assess_evidence("What should I check for KD-401?", [chunk("For KD-401, check that indexing is enabled.")])
    assert decision.answerability == "answerable"
