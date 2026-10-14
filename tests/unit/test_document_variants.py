import json
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.loaders import PageText, load_document
from app.ingestion.chunker import chunk_pages
from app.ingestion.profile import analyze
from app.ingestion.structure import source_units
from app.generation.grounding import passage_key, supporting_passages, validate_selection
from app.retrieval.fusion import RetrievedChunk


@pytest.mark.parametrize("extension", [".txt", ".text", ".md", ".markdown"])
@pytest.mark.parametrize("content", [b"", b" \n\t ", b"\x00\x01binary"])
def test_empty_and_binary_text_rejected(tmp_path, extension, content):
    path = tmp_path / ("empty" + extension)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        load_document(path, extension)


def test_unicode_and_bom_are_preserved(tmp_path):
    path = tmp_path / "utf16.txt"
    path.write_bytes("The caf\u00e9 closes at 18:00. Invoice total: \u20ac25.".encode("utf-16"))
    assert "caf\u00e9" in load_document(path, ".txt")[0].text


def test_invalid_text_encoding_is_not_silently_replaced(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"bad\xffencoding")
    with pytest.raises(ValueError, match="UTF"):
        load_document(path, ".txt")


@pytest.mark.parametrize("extension,content", [(".pdf", b"broken PDF"), (".docx", b"not a zip"), (".json", b"{broken")])
def test_corrupt_files_fail_clearly(tmp_path, extension, content):
    path = tmp_path / ("bad" + extension)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        load_document(path, extension)


@pytest.mark.parametrize("content", ["", "Name,Price\n", "Name,Name\nA,12\n", "Name,Price\nA,12,wrong\n"])
def test_ambiguous_or_empty_csv_fails(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(path, ".csv")


def test_csv_rows_keep_column_labels_and_zero_values(tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text("Plan,Monthly USD,Annual USD\nStarter,0,0\nPro,25,250\n", encoding="utf-8")
    page = load_document(path, ".csv")[0]
    units = source_units(page.text)
    assert len(units) == 2
    assert all(u.kind == "table_row" for u in units)
    row = page.text[units[1].start:units[1].end]
    assert '"Monthly USD": "25"' in row and '"Annual USD": "250"' in row
    assert '"Monthly USD": "0"' in page.text


def test_nested_json_preserves_key_path_and_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"limits": {"retry": 0}, "enabled": False}), encoding="utf-8")
    page = load_document(path, ".json")[0]
    assert "root.limits.retry: 0" in page.text
    assert "root.enabled: false" in page.text


@pytest.mark.parametrize("size,overlap", [(12, 2), (25, 5), (40, 8)])
def test_token_chunks_cover_every_word_and_keep_exact_offsets(size, overlap):
    text = " ".join(f"Sentence {i} contains information about item {i}." for i in range(20))
    page = PageText(text, 2)
    chunks = chunk_pages("doc", [page], target_tokens=size, overlap_tokens=overlap,
                         token_counter=lambda value: len(value.split()))
    assert chunks
    assert all(chunk.token_count <= size for chunk in chunks)
    assert all(page.text[chunk.start_offset:chunk.end_offset] == chunk.text for chunk in chunks)
    for i in range(20):
        assert any(f"item {i}" in chunk.text for chunk in chunks)


def test_long_single_sentence_is_bounded_but_source_unit_is_preserved():
    text = "This clause " + "word " * 200 + "ends here."
    chunks = chunk_pages("doc", [PageText(text, 1)], target_tokens=30, overlap_tokens=5,
                         token_counter=lambda value: len(value.split()))
    assert len(chunks) > 1
    assert len(source_units(text, pdf=True)) == 1
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_abbreviations_and_decimals_do_not_split_facts():
    text = "Dr. Rao paid Rs. 25.50 for the item. The refund period is 30 days."
    units = source_units(text)
    assert len(units) == 2
    assert "25.50" in text[units[0].start:units[0].end]


def test_pdf_columns_and_blank_pages_keep_page_identity(tmp_path):
    path = tmp_path / "columns.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 80, 260, 250), "The warranty period is 24 months.\nDefective products may be returned within 30 days.")
    page.insert_textbox(pymupdf.Rect(310, 80, 540, 250), "The invoice total is 75 dollars.\nStandard delivery takes 4 days.")
    doc.new_page()
    doc.save(path)
    doc.close()
    pages = load_document(path, ".pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 1 and pages[1].page_number == 2
    assert "24 months" in pages[0].text and "75 dollars" in pages[0].text
    analysis = analyze(pages, lambda value: len(value.split()), 200, 40, 256)
    assert analysis["empty_pages"] == [2]


def test_password_protected_pdf_is_rejected(tmp_path):
    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Private contract.")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
    doc.close()
    with pytest.raises(ValueError, match="Password"):
        load_document(path, ".pdf")


def test_image_only_pdf_reports_ocr_requirement(tmp_path, monkeypatch):
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path.name == "eng.traineddata" else original(path))
    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pixmap.clear_with(255)
    page.insert_image(pymupdf.Rect(20, 20, 120, 120), pixmap=pixmap)
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError, match="OCR"):
        load_document(path, ".pdf")


def test_image_only_pdf_rejects_scans_even_with_bundled_ocr(tmp_path):
    source = pymupdf.open()
    page = source.new_page(width=500, height=150)
    page.insert_text((25, 60), "Defective items can be returned within 30 days.", fontsize=16)
    image = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
    raster = pymupdf.open()
    page = raster.new_page(width=500, height=150)
    page.insert_image(page.rect, pixmap=image)
    path = tmp_path / "ocr.pdf"
    raster.save(path)
    raster.close()
    source.close()
    with pytest.raises(ValueError, match="OCR is not enabled"):
        load_document(path, ".pdf")


def test_semantic_paraphrase_can_use_verified_source_text():
    quote = "Defective merchandise can be returned within thirty days."
    item = RetrievedChunk("c", "d", "returns.txt", None, quote, 1.0, "test",
                          {"passage_similarities": {passage_key(quote): 0.8}})
    passages = supporting_passages("How long do I have to send a broken item back?", [item])
    assert passages
    answer, citations, metrics = validate_selection('{"passage_ids": [1]}', passages)
    assert quote in answer and citations[0].excerpt == quote
    assert metrics["faithfulness"] == 1.0
