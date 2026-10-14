import json

from app.generation.grounding import ABSTENTION, Passage, supporting_passages, validate_selection
from app.retrieval.fusion import RetrievedChunk


def chunk(text, chunk_id="c1"):
    return RetrievedChunk(chunk_id, "d1", "guide.txt", None, text, 1.0, "test")


def test_filename_does_not_answer_personal_name():
    assert not supporting_passages("What is my name?", [chunk("The filename field contains the name of the uploaded document.")])


def test_question_in_document_is_not_an_answer():
    assert not supporting_passages("What is the office address?", [chunk("What is the office address?\nThe API supports PDF files.")])


def test_claim_links_to_exact_supporting_passage_only():
    evidence = [chunk("For a 401 error, verify the bearer token. The API supports PDF files.")]
    passages = supporting_passages("How do I fix a 401 error?", evidence)
    answer, citations, metrics = validate_selection('{"passage_ids": [1]}', passages)
    assert len(citations) == 1
    assert citations[0].excerpt == "For a 401 error, verify the bearer token."
    assert "PDF" not in answer
    assert f"[1](#{citations[0].anchor})" in answer
    assert metrics["faithfulness"] == 1.0


def test_unknown_citation_fails_closed_and_measures_rejected_claim():
    passages = supporting_passages("401 error", [chunk("For a 401 error, verify the bearer token.")])
    answer, citations, metrics = validate_selection('{"passage_ids": [1, 999]}', passages)
    assert answer == ABSTENTION
    assert not citations
    assert metrics["selection_precision"] == 0.5
    assert metrics["faithfulness"] is None


def test_freeform_hallucination_is_rejected():
    passages = supporting_passages("401 error", [chunk("For a 401 error, verify the bearer token.")])
    answer, citations, metrics = validate_selection("Ensure that your token is missing or invalid.", passages)
    assert answer == ABSTENTION
    assert citations == []


def test_unavailable_information_returns_one_line():
    answer, citations, _ = validate_selection(json.dumps({"passage_ids": []}), [])
    assert answer == ABSTENTION
    assert "\n" not in answer
    assert not citations


def test_upload_limit_cannot_use_unrelated_search_limit():
    evidence = [chunk("The default search limit is 25. The default upload size limit is 20 MB.")]
    passages = supporting_passages("What is the default upload size limit?", evidence)
    assert passages
    assert all("upload" in p.quote for p in passages)


def test_maximum_limit_is_not_automatically_a_default_limit():
    assert not supporting_passages("What is the default upload size limit?",
                                   [chunk("The maximum upload size limit is 20 MB.")])


def test_default_limit_keeps_its_endpoint_context():
    evidence = [chunk("The API search endpoint accepts a query and limit parameter. The default limit is 25.")]
    passages = supporting_passages("What is the default API search limit?", evidence)
    assert any("25" in p.quote and "search" in p.quote for p in passages)


def test_contained_overlap_quotes_produce_one_citation():
    text = "The warranty lasts 24 months. It includes parts and labour."
    item = chunk(text)
    passages = [Passage(1, item, "The warranty lasts 24 months."), Passage(2, item, text)]
    answer, citations, metrics = validate_selection('{"passage_ids": [1, 2]}', passages)
    assert len(citations) == 1 and citations[0].excerpt == text
    assert answer.count("24 months") == 1
    assert metrics["supported_claims"] == 1
