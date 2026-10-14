import pytest
from app.api.routes import chat as chat_routes
from app.core.errors import RAGServiceError
from tests.conftest import signup, auth_headers


@pytest.mark.parametrize("extension", ["txt", "md"])
def test_query_endpoint_upload_cites_and_deletes(client, extension):
    from app.retrieval.vector_store import ChromaVectorStore
    headers = auth_headers(signup(client)["access_token"])
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    fact = "The Citrine press warranty period is 29 months."
    text = ("# Service Warranty\n" if extension == "md" else "") + fact + "\n\n" + "\n".join(
        f"Maintenance record {i}: Inspect the inlet filter and clean the housing before starting normal operation each day."
        for i in range(45))
    upload = client.post("/documents/upload", headers=headers,
                         files={"file": (f"fresh-citrine.{extension}", text.encode())})
    assert upload.status_code == 200
    assert upload.json()["analysis"]["chunk_count"] > 1
    assert upload.json()["analysis"]["max_chunk_tokens"] <= 240
    assert upload.json()["analysis"]["overlap_tokens"] == 60
    doc = upload.json()["document_id"]
    store = ChromaVectorStore()
    stored = store._collection.get(where={"document_id": doc}, include=["documents", "metadatas"])
    assert len(stored["ids"]) == upload.json()["analysis"]["chunk_count"]
    assert all(m["source"] == f"fresh-citrine.{extension}" and m["document_id"] == doc for m in stored["metadatas"])
    assert any(d["document_id"] == doc for d in client.get("/documents", headers=headers).json())
    response = client.post("/chat/query", headers=headers,
                           json={"question": "What is the warranty period for the Citrine press?", "document_ids": [doc]})
    assert response.status_code == 200
    citation = response.json()["citations"][0]
    assert citation["source_filename"] == f"fresh-citrine.{extension}"
    assert citation["document_id"] == doc and citation["chunk_id"] in stored["ids"]
    assert citation["excerpt"] == fact and "29 months" in response.json()["answer"]
    if extension == "md":
        assert citation["section"] == "Service Warranty"
    unrelated = client.post("/chat/query", headers=headers,
                            json={"question": "Who won the football match?", "document_ids": [doc]})
    assert unrelated.status_code == 200
    assert unrelated.json()["answer"] == "I could not find enough information in the uploaded document to answer this."
    assert unrelated.json()["citations"] == []
    assert client.delete(f"/documents/{doc}", headers=headers).status_code == 200
    assert not store._collection.get(where={"document_id": doc})["ids"]


@pytest.mark.parametrize("code", ["OLLAMA_UNAVAILABLE", "MODEL_NOT_INSTALLED", "EMBEDDING_UNAVAILABLE"])
def test_query_service_errors_are_not_answers(client, monkeypatch, code):
    headers = auth_headers(signup(client)["access_token"])
    def fail(*args):
        raise RAGServiceError(code, "Service requires setup")
    monkeypatch.setattr(chat_routes, "ensure_ollama_model", fail)
    response = client.post("/chat/query", headers=headers, json={"question": "Question"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == code
    assert "answer" not in response.json()


def test_upload_embedding_failure_is_503(client, monkeypatch):
    from app.ingestion import pipeline
    headers = auth_headers(signup(client)["access_token"])
    def fail(*args):
        raise RAGServiceError("EMBEDDING_UNAVAILABLE", "Embedding model unavailable")
    monkeypatch.setattr(pipeline, "shared_embedding_model", fail)
    response = client.post("/documents/upload", headers=headers, files={"file": ("guide.txt", b"A real document.")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EMBEDDING_UNAVAILABLE"


def test_query_model_allowlist_and_unrelated_abstention(client):
    headers = auth_headers(signup(client)["access_token"])
    assert client.post("/chat/query", headers=headers, json={"question": "Question", "model_name": "llama3:8b"}).status_code == 422
    response = client.post("/chat/query", headers=headers, json={"question": "Who won the football match?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "I could not find enough information in the uploaded document to answer this."
    assert response.json()["citations"] == []
