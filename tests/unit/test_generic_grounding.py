from app.generation.grounding import supporting_passages
from app.ingestion.structure import source_units
from app.retrieval.fusion import RetrievedChunk


def document(text, filename="manual.pdf"):
    units = [unit.__dict__ for unit in source_units(text, pdf=True)]
    return [RetrievedChunk("chunk", "doc", filename, 1, text, 1, "test", {
        "page_source_text": text, "source_units": units,
    })]


def test_question_verb_is_not_a_mandatory_source_word():
    text = (
        "During Q3:2025, equipment settings were reviewed. The RPM increased by 5 units. "
        "The change in the warranty period was discussed separately."
    )
    passages = supporting_passages("How did RPM change during Q3:2025?", document(text))
    assert any("increased by 5 units" in p.quote for p in passages)


def test_period_cannot_be_borrowed_from_an_unrelated_paragraph():
    text = (
        "The RPM increased by 5 units during Q2:2025.\n\n"
        "During Q3:2025, warranty conditions were reviewed."
    )
    assert not supporting_passages("How did RPM change during Q3:2025?", document(text))


def test_monthly_row_cannot_answer_whole_cycle_question():
    text = 'ROW 1: {"Period": "January 2025", "RPM": "5"}'
    assert not supporting_passages("How did RPM change in the current maintenance cycle?", document(text))


def test_pdf_sentence_with_footnote_marker_is_complete():
    text = "The maximum service interval was recorded for Orion equipment.13"
    passages = supporting_passages("Which equipment had the maximum service interval?", document(text))
    assert passages and "Orion" in passages[0].quote


def test_explicit_negative_support_is_an_answer():
    text = "The export feature supports CSV and JSON. XML export is not supported."
    passages = supporting_passages("Does the export feature support XML?", document(text, "guide.txt"))
    assert any("XML export is not supported" in p.quote for p in passages)


def test_default_size_can_be_described_as_configured_maximum():
    text = "The maximum supported upload size in the default configuration is 20 MB."
    passages = supporting_passages("What is the default upload size limit?", document(text, "faq.txt"))
    assert passages and "20 MB" in passages[0].quote
