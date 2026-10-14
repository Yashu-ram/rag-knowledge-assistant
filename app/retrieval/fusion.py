from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int | None
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    metadata: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            by_id[item.chunk_id] = item
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(item.chunk_id, set()).add(item.source)
            metadata.setdefault(item.chunk_id, {}).update(item.metadata)
    fused = []
    for chunk_id, item in by_id.items():
        merged = RetrievedChunk(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            source_filename=item.source_filename,
            page_number=item.page_number,
            text=item.text,
            score=scores[chunk_id],
            source="+".join(sorted(sources[chunk_id])),
            metadata=metadata[chunk_id],
        )
        fused.append(merged)
    return sorted(fused, key=lambda item: item.score, reverse=True)

