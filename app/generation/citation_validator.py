from app.api.schemas import Citation
from app.retrieval.fusion import RetrievedChunk


def build_citations(evidence: list[RetrievedChunk], max_citations: int = 4) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for item in evidence:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        citations.append(
            Citation(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_filename=item.source_filename,
                page_number=item.page_number,
                excerpt=item.text[:700],
            )
        )
        if len(citations) >= max_citations:
            break
    return citations

