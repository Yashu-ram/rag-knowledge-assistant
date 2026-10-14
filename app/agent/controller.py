from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agent.tools import RetrievalSettings, RetrievalTools, ToolResult
from app.retrieval.evidence_policy import EvidenceDecision, assess_evidence
from app.retrieval.fusion import RetrievedChunk
from app.retrieval.keyword_search import tokenize


@dataclass(frozen=True)
class AgentResult:
    evidence: list[RetrievedChunk]
    answerability: str
    tool_calls: list[dict]
    retry_count: int
    reason: str


class RetrievalAgent:
    def __init__(self, tools: RetrievalTools | None = None, max_retries: int = 1) -> None:
        self.tools = tools or RetrievalTools()
        self.max_retries = max_retries

    def run(
        self,
        db: Session,
        user_id: str,
        question: str,
        settings: RetrievalSettings,
        conversation_context: list[str] | None = None,
    ) -> AgentResult:
        tool_calls: list[dict] = []
        query = self._build_query(question, conversation_context)

        first = self.tools.hybrid_search(db, user_id, query, settings)
        tool_calls.append(self._summarize_tool(first))
        reranked = self.tools.rerank_evidence(question, first.results)
        tool_calls.append(self._summarize_tool(reranked))

        decision = assess_evidence(question, reranked.results)
        retry_count = 0
        evidence = reranked.results

        if decision.answerability == "unanswerable" and retry_count < self.max_retries:
            retry_count += 1
            retry_query = self._retry_query(question)
            retry = self.tools.hybrid_search(db, user_id, retry_query, settings)
            tool_calls.append(self._summarize_tool(retry))
            retry_reranked = self.tools.rerank_evidence(question, retry.results)
            tool_calls.append(self._summarize_tool(retry_reranked))
            retry_decision = assess_evidence(question, retry_reranked.results)
            if self._better(retry_decision, decision):
                decision = retry_decision
                evidence = retry_reranked.results

        return AgentResult(
            evidence=evidence,
            answerability=decision.answerability,
            tool_calls=tool_calls,
            retry_count=retry_count,
            reason=decision.reason,
        )

    @staticmethod
    def _build_query(question: str, conversation_context: list[str] | None) -> str:
        if not conversation_context:
            return question
        recent_context = " ".join(conversation_context[-2:])
        return f"{recent_context} {question}".strip()

    @staticmethod
    def _retry_query(question: str) -> str:
        return " ".join(tokenize(question))

    @staticmethod
    def _summarize_tool(result: ToolResult) -> dict:
        return {
            "tool": result.tool_name,
            "query": result.query,
            "result_count": len(result.results),
            "top_chunk_id": result.results[0].chunk_id if result.results else None,
            "diagnostics": result.diagnostics,
        }

    @staticmethod
    def _better(candidate: EvidenceDecision, current: EvidenceDecision) -> bool:
        order = {"unanswerable": 0, "partial": 1, "answerable": 2}
        return order[candidate.answerability] > order[current.answerability]

