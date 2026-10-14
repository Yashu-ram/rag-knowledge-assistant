from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import PageText


def test_chunk_pages_preserves_overlap_and_page_number():
    chunks = chunk_pages("doc1", [PageText(text="abcdefghijklmnopqrstuvwxyz", page_number=3)], chunk_size=10, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].page_number == 3
    assert chunks[1].start_offset == 8

