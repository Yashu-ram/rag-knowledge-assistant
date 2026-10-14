import pytest

from app.generation.grounding import ABSTENTION
from app.main import app
from app.storage.database import get_db
from app.storage.models import Document
from tests.conftest import auth_headers, signup
from tests.integration.test_rag_api import upload_text


def test_overview_uses_representative_chunks_and_cited_bullets(client):
    user = signup(client)
    token = user["access_token"]
    content = "\n\n".join(
        f"{topic} policy. " + " ".join(
            f"{topic} rule {i} requires staff to record the request and complete verification within {i + 2} days."
            for i in range(18)
        )
        for topic in ["Delivery", "Returns", "Billing", "Support"]
    )
    document = upload_text(client, token, "policies.md", content)
    upload_text(client, token, "other.txt", "Unrelated private notes about astronomy.")
    response = client.post("/chat/query", headers=auth_headers(token), json={
        "question": "Summarize the uploaded document using citations",
        "document_ids": [document["document_id"]],
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answerable"] == "answerable"
    assert 3 <= len(data["answer"].splitlines()) <= 5
    assert all(line.startswith("- ") for line in data["answer"].splitlines())
    assert len({c["chunk_id"] for c in data["citations"]}) > 1
    for citation in data["citations"]:
        assert citation["document_id"] == document["document_id"]
        assert citation["excerpt"] in content
        assert f"[{citation['citation_id']}](#{citation['anchor']})" in data["answer"]
    metadata = data["retrieval_metadata"]
    assert metadata["overview"] is True
    diagnostics = metadata["tool_calls"][0]["diagnostics"]
    assert len(diagnostics["first_chunk_ids"]) == 2
    assert 4 <= len(diagnostics["semantic_chunk_ids"]) <= 8
    assert diagnostics["keyword_chunk_ids"]
    retrieved_ids = [c["chunk_id"] for c in metadata["retrieved_chunks"]]
    assert len(retrieved_ids) == len(set(retrieved_ids))
    assert set(diagnostics["first_chunk_ids"]) <= set(retrieved_ids)
    assert len(retrieved_ids) <= 6


def test_ipl_question_still_abstains_without_generation(client, monkeypatch):
    from app.api.routes import chat as chat_routes

    user = signup(client)
    document = upload_text(client, user["access_token"], "policies.txt", "Delivery takes five days. Returns are free for thirty days.")

    def unexpected_generation(*args):
        pytest.fail("Unrelated factual question must not call the LLM")

    monkeypatch.setattr(chat_routes, "generate_answer", unexpected_generation)
    response = client.post("/chat/query", headers=auth_headers(user["access_token"]), json={
        "question": "What is today's IPL score?", "document_ids": [document["document_id"]],
    })
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == ABSTENTION
    assert response.json()["citations"] == []
    assert response.json()["retrieval_metadata"]["overview"] is False


@pytest.mark.parametrize("status", ["processing", "failed"])
def test_overview_rejects_incomplete_document(client, status):
    user = signup(client)
    document = upload_text(client, user["access_token"], "pending.txt", "Delivery takes five days.")
    dependency = app.dependency_overrides[get_db]()
    db = next(dependency)
    try:
        db.get(Document, document["document_id"]).status = status
        db.commit()
    finally:
        dependency.close()
    response = client.post("/chat/query", headers=auth_headers(user["access_token"]), json={
        "question": "What is this document about?", "document_ids": [document["document_id"]],
    })
    assert response.status_code == 409, response.text


def test_overview_empty_index_abstains(client):
    user = signup(client)
    response = client.post("/chat/query", headers=auth_headers(user["access_token"]), json={
        "question": "Give me the key points",
    })
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == ABSTENTION
    assert response.json()["citations"] == []
