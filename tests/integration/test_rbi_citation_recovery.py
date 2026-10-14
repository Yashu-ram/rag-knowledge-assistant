from pathlib import Path
from unittest.mock import Mock
import json

import pytest

from app.api.routes import chat
from tests.conftest import auth_headers, signup

PDF = Path(__file__).resolve().parents[2] / "sample_documents/RBI_Policy_RAG_Sample.pdf"


@pytest.mark.parametrize("question,text,value", [
    ("What was the cumulative policy rate reduction during the 2025 easing cycle?",
     "The cumulative policy rate reduction was 125 bps.", "125"),
    ("How did MCLR change in H2:2025-26?", "MCLR declined by 20 bps in H2:2025-26.", "20"),
])
def test_rbi_pdf_answer_without_citation_gets_used_chunk_support(client, monkeypatch, question, text, value):
    headers = auth_headers(signup(client)["access_token"])
    result = client.post("/documents/upload", headers=headers, files={"file": (PDF.name, PDF.read_bytes())})
    assert result.status_code == 200, result.text
    document = result.json()
    used_chunks = []
    def generator(_, evidence, *__):
        used_chunks.extend(item.chunk_id for item in evidence)
        return json.dumps({"claims": [{"text": text}]}), "test-model", 1
    monkeypatch.setattr(chat, "generate_answer", generator)
    response = client.post("/chat/query", headers=headers,
                           json={"question": question, "document_id": document["document_id"], "model": "qwen2.5:1.5b"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answerable"] == "answerable" and value in data["answer"]
    assert data["citations"]
    assert data["retrieval_metadata"]["grounding"]["citations_recovered"] >= 1
    for citation in data["citations"]:
        assert citation["source_filename"] == PDF.name
        assert citation["document_id"] == document["document_id"]
        assert citation["chunk_id"] in used_chunks
        assert citation["page_number"] in {1, 2}
        assert value in citation["excerpt"]


def test_rbi_pdf_unrelated_question_still_abstains(client, monkeypatch):
    headers = auth_headers(signup(client)["access_token"])
    document = client.post("/documents/upload", headers=headers, files={"file": (PDF.name, PDF.read_bytes())}).json()
    generator = Mock(return_value=('{"claims": []}', "test-model", 1))
    monkeypatch.setattr(chat, "generate_answer", generator)
    response = client.post("/chat/query", headers=headers,
                           json={"question": "What is today's IPL score?", "document_ids": [document["document_id"]]})
    assert response.status_code == 200
    assert response.json()["answerable"] == "unanswerable"
    assert not response.json()["citations"]


def test_single_document_alias_preserves_ownership_checks(client):
    headers = auth_headers(signup(client)["access_token"])
    response = client.post("/chat/query", headers=headers,
                           json={"question": "Test?", "document_id": "missing-document", "model": "qwen2.5:1.5b"})
    assert response.status_code == 404
