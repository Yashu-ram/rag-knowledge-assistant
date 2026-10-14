from app.retrieval.vector_store import ChromaVectorStore
from app.storage.models import Chunk


def test_chroma_metadata_uses_grounding_fields():
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        user_id="u1",
        source_filename="guide.md",
        page_number=None,
        chunk_index=3,
        start_offset=0,
        end_offset=10,
        text="hello world",
        embedding_json="[0.1]",
    )
    metadata = ChromaVectorStore._metadata(chunk)
    assert metadata["user_id"] == "u1"
    assert metadata["document_id"] == "d1"
    assert metadata["source_filename"] == "guide.md"
    assert metadata["page_number"] == -1
    assert metadata["chunk_index"] == 3

