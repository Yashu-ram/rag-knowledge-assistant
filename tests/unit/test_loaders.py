from app.ingestion.loaders import load_document
from zipfile import ZipFile


def test_text_extension_loads_as_plain_text(tmp_path):
    path = tmp_path / "notes.text"
    path.write_text("KiwiDesk supports grounded answers from text files.", encoding="utf-8")
    pages = load_document(path, ".text")
    assert len(pages) == 1
    assert "grounded answers" in pages[0].text
    assert pages[0].page_number is None


def test_docx_loads_text(tmp_path):
    path = tmp_path / "notes.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>HireKiwi indexes Word documents.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Answers still cite uploaded sources.</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    pages = load_document(path, ".docx")

    assert len(pages) == 1
    assert "Word documents" in pages[0].text
    assert "uploaded sources" in pages[0].text

