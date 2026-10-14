"""End-to-end ingestion and citation tests across formats and failure states."""

import pymupdf
import pytest

from tests.conftest import auth_headers, signup


def upload(client, token, name, content):
    return client.post("/documents/upload", headers=auth_headers(token), files={"file": (name, content)})


@pytest.mark.parametrize("name,content", [
    ("empty.txt", b""), ("blank.md", b" \n\t"), ("bad.pdf", b"not PDF"),
    ("bad.docx", b"not ZIP"), ("bad.json", b"{bad"), ("video.mp4", b"video"),
])
def test_invalid_uploads_never_create_searchable_chunks(client, name, content):
    user = signup(client)
    result = upload(client, user["access_token"], name, content)
    assert result.status_code == 400
    chat = client.post("/chat", headers=auth_headers(user["access_token"]), json={"question": "What is the warranty period?"})
    assert chat.json()["citations"] == []
    assert chat.json()["answerable"] == "unanswerable"


def test_index_profile_and_duplicate_upload_are_stable(client):
    user = signup(client)
    content = b"The warranty period is 24 months. Standard delivery takes 4 days."
    first = upload(client, user["access_token"], "manual.txt", content).json()
    second = upload(client, user["access_token"], "manual.txt", content).json()
    assert first["document_id"] == second["document_id"]
    assert first["analysis"]["index_version"] == "langchain-structured-v5"
    assert first["analysis"]["max_chunk_tokens"] <= first["analysis"]["target_tokens"]


@pytest.mark.parametrize("name,content,question,expected", [
    ("warranty.txt", b"The warranty period is 24 months.", "What is the warranty period?", "24 months"),
    ("contract.md", b"# Contract\nEither party must give 60 days of notice before terminating the agreement.", "How many days of notice are needed before terminating the agreement?", "60 days"),
    ("prices.csv", b"Plan,Monthly USD,Annual USD\nStarter,12,120\nPro,25,250\n", "What is the Monthly USD fee for Pro?", "25"),
    ("settings.json", b'{"limits":{"retry":3}}', "What is the retry limit?", "3"),
])
def test_varied_domains_and_formats_have_verifiable_citations(client, name, content, question, expected):
    user = signup(client)
    document = upload(client, user["access_token"], name, content)
    assert document.status_code == 200, document.text
    answer = client.post("/chat", headers=auth_headers(user["access_token"]),
                         json={"question": question, "document_ids": [document.json()["document_id"]]}).json()
    assert answer["citations"], answer
    assert expected in answer["answer"]
    assert answer["retrieval_metadata"]["grounding"]["faithfulness"] == 1.0


def test_reindex_failure_keeps_the_previous_working_index(client, monkeypatch):
    from app.ingestion import pipeline
    user = signup(client)
    document = upload(client, user["access_token"], "warranty.txt", b"The warranty period is 24 months.").json()
    monkeypatch.setattr(pipeline.StructuredDocumentLoader, "load", lambda *args: (_ for _ in ()).throw(ValueError("Parser failed")))
    response = client.post(f"/documents/{document['document_id']}/reindex", headers=auth_headers(user["access_token"]))
    assert response.status_code == 400
    stored = client.get(f"/documents/{document['document_id']}", headers=auth_headers(user["access_token"])).json()
    assert stored["status"] == "indexed"
    response = client.post("/chat", headers=auth_headers(user["access_token"]), json={"question": "What is the warranty period?"})
    assert "24 months" in response.json()["answer"]


def test_reindex_does_not_duplicate_chunks(client):
    user = signup(client)
    document = upload(client, user["access_token"], "manual.txt", b"The warranty period is 24 months.").json()
    response = client.post(f"/documents/{document['document_id']}/reindex", headers=auth_headers(user["access_token"]))
    assert response.status_code == 200, response.text
    assert response.json()["analysis"]["chunk_count"] == document["analysis"]["chunk_count"]


def test_source_unit_expansion_recovers_fact_beyond_embedding_boundary(client, monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "chunk_target_tokens", 64)
    monkeypatch.setattr(get_settings(), "chunk_overlap_tokens", 12)
    user = signup(client)
    text = "The permitted exchange window under this policy " + "includes conditions " * 80 + "and lasts 45 days."
    document = upload(client, user["access_token"], "long.txt", text.encode()).json()
    assert document["analysis"]["chunk_count"] > 1
    answer = client.post("/chat", headers=auth_headers(user["access_token"]),
                         json={"question": "What is the permitted exchange window under this policy?"}).json()
    assert answer["citations"], answer
    assert "45 days" in answer["answer"]
    assert answer["citations"][0]["excerpt"] == text
