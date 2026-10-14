from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.retrieval.fusion import RetrievedChunk
from app.retrieval.service import RetrievalService
from app.generation.grounding import terms


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 6
    document_ids: list[str] | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    query: str
    results: list[RetrievedChunk]
    diagnostics: dict = field(default_factory=dict)


class RetrievalTools:
    def __init__(self, retrieval: RetrievalService | None = None) -> None:
        self.retrieval = retrieval or RetrievalService()

    def semantic_search(self, db: Session, user_id: str, query: str, settings: RetrievalSettings) -> ToolResult:
        results = self.retrieval.semantic_search(db, user_id, query, settings.top_k, settings.document_ids)
        return ToolResult(tool_name="semantic_search", query=query, results=results)

    def keyword_search(self, db: Session, user_id: str, query: str, settings: RetrievalSettings) -> ToolResult:
        results = self.retrieval.keyword_search(db, user_id, query, settings.top_k, settings.document_ids)
        return ToolResult(tool_name="keyword_search", query=query, results=results)

    def hybrid_search(self, db: Session, user_id: str, query: str, settings: RetrievalSettings) -> ToolResult:
        results = self.retrieval.hybrid_search(db, user_id, query, settings.top_k, settings.document_ids)
        return ToolResult(tool_name="hybrid_search", query=query, results=results,
                          diagnostics=dict(self.retrieval.search_diagnostics))

    def rerank_evidence(self, query: str, candidates: list[RetrievedChunk]) -> ToolResult:
        # Deterministic lightweight rerank: combine fused score with query-term coverage.
        query_terms = terms(query)

        def score(item: RetrievedChunk) -> float:
            coverage = len(query_terms & terms(item.text))
            return item.score + (coverage * 0.05)

        results = sorted(candidates, key=score, reverse=True)
        return ToolResult(tool_name="rerank_evidence", query=query, results=results)

