from app.agent.controller import RetrievalAgent
from app.agent.tools import RetrievalSettings, ToolResult
from app.retrieval.fusion import RetrievedChunk


class FakeTools:
    def __init__(self) -> None:
        self.calls = []

    def hybrid_search(self, db, user_id, query, settings):
        self.calls.append(("hybrid", query))
        return ToolResult("hybrid_search", query, [])

    def rerank_evidence(self, query, candidates):
        self.calls.append(("rerank", query))
        return ToolResult("rerank_evidence", query, candidates)


class AnswerableTools(FakeTools):
    def hybrid_search(self, db, user_id, query, settings):
        self.calls.append(("hybrid", query))
        result = RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            source_filename="doc.md",
            page_number=None,
            text="KiwiDesk release version is 4.2 and Smart Inbox is supported.",
            score=1.0,
            source="hybrid",
        )
        result2 = RetrievedChunk(
            chunk_id="c2",
            document_id="d1",
            source_filename="doc.md",
            page_number=None,
            text="The current supported release is KiwiDesk 4.2.",
            score=0.9,
            source="hybrid",
        )
        return ToolResult("hybrid_search", query, [result, result2])


def test_agent_retries_once_when_unanswerable():
    tools = FakeTools()
    agent = RetrievalAgent(tools=tools, max_retries=1)
    result = agent.run(None, "u1", "Who is CEO?", RetrievalSettings())
    assert result.answerability == "unanswerable"
    assert result.retry_count == 1


def test_agent_returns_answerable_evidence():
    agent = RetrievalAgent(tools=AnswerableTools(), max_retries=1)
    result = agent.run(None, "u1", "What KiwiDesk release version is supported?", RetrievalSettings())
    assert result.answerability == "answerable"
    assert len(result.evidence) == 2

