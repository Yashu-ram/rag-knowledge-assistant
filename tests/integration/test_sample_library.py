import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select

from app.ingestion.samples import ingest_sample_library
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import ChromaVectorStore
from app.storage.database import get_db
from app.storage.document_access import SAMPLE_LIBRARY_USER_ID
from app.storage.models import Document, User
from tests.conftest import auth_headers, signup
from tests.integration.test_rag_api import upload_text

SAMPLES = {"demo.txt": "The Orion press warranty lasts 31 months.",
           "nested/guide.md": "# Orion maintenance\nThe Orion press needs inspection every 7 days."}
BUNDLED_SAMPLES = {path.name: path.read_bytes()
                   for path in (Path(__file__).resolve().parents[2] / "sample_documents").iterdir()
                   if path.is_file()}


@pytest.mark.parametrize("client", [BUNDLED_SAMPLES], indirect=True)
def test_all_bundled_files_including_pdf_are_indexed_at_startup(client):
    token = signup(client)["access_token"]
    response = client.get("/documents", headers=auth_headers(token))
    assert response.status_code == 200
    docs = response.json()
    assert {doc["filename"] for doc in docs} == set(BUNDLED_SAMPLES)
    assert all(doc["is_sample"] and doc["status"] == "indexed" for doc in docs), docs
    assert all(doc["analysis"]["chunk_count"] > 0 for doc in docs)


@pytest.mark.parametrize("client", [SAMPLES], indirect=True)
def test_startup_samples_visible_and_queryable_for_new_and_returning_users(client):
    first = signup(client, email="first@example.com")
    second = signup(client, email="second@example.com")
    listed = client.get("/documents", headers=auth_headers(first["access_token"])).json()
    assert {doc["filename"] for doc in listed} == {"demo.txt", "guide.md"}
    assert all(doc["is_sample"] and doc["status"] == "indexed" for doc in listed)
    assert all(doc["analysis"]["chunk_count"] > 0 for doc in listed)
    assert client.get("/documents", headers=auth_headers(second["access_token"])).json() == listed
    doc_id = next(doc["document_id"] for doc in listed if doc["filename"] == "demo.txt")
    for account in (first, second):
        headers = auth_headers(account["access_token"])
        assert client.get(f"/documents/{doc_id}", headers=headers).status_code == 200
        for selection in ([doc_id], None):
            result = client.post("/chat/query", headers=headers, json={
                "question": "What is the Orion press warranty period?", "document_ids": selection})
            assert result.status_code == 200, result.text
            assert "31 months" in result.json()["answer"]
            assert result.json()["citations"][0]["document_id"] == doc_id
        overview = client.post("/chat/query", headers=headers, json={
            "question": "Summarize this document", "document_ids": [doc_id]})
        assert overview.status_code == 200, overview.text
        assert overview.json()["citations"]
    login = client.post("/auth/login", json={"email": "second@example.com", "password": "strong-password-123"})
    assert login.status_code == 200
    assert client.get("/documents", headers=auth_headers(login.json()["access_token"])).json() == listed
    assert client.get("/documents").status_code == 401


@pytest.mark.parametrize("client", [SAMPLES], indirect=True)
def test_shared_samples_are_read_only_and_personal_uploads_stay_private(client):
    first = signup(client, email="first@example.com")
    second = signup(client, email="second@example.com")
    first_headers, second_headers = auth_headers(first["access_token"]), auth_headers(second["access_token"])
    samples = client.get("/documents", headers=second_headers).json()
    private = upload_text(client, first["access_token"], "private.txt", "The secret vault code is 98765.")
    assert not private["is_sample"]
    assert client.get("/documents", headers=second_headers).json() == samples
    for doc in samples:
        assert client.delete(f"/documents/{doc['document_id']}", headers=second_headers).status_code == 404
        assert client.post(f"/documents/{doc['document_id']}/reindex", headers=second_headers).status_code == 404
    private_id = private["document_id"]
    assert client.get(f"/documents/{private_id}", headers=second_headers).status_code == 404
    assert client.post("/chat/query", headers=second_headers, json={
        "question": "What is the secret vault code?", "document_ids": [private_id]}).status_code == 404
    result = client.post("/chat/query", headers=second_headers, json={"question": "What is the secret vault code?"})
    assert result.status_code == 200, result.text
    assert "98765" not in result.json()["answer"]
    assert all(c["document_id"] != private_id for c in result.json()["citations"])
    assert client.delete(f"/documents/{private_id}", headers=first_headers).status_code == 200


@pytest.mark.parametrize("client", [SAMPLES], indirect=True)
def test_restarting_library_reuses_ids_and_both_search_paths_include_samples(client):
    from app.main import app

    account = signup(client)
    headers = auth_headers(account["access_token"])
    before = client.get("/documents", headers=headers).json()
    dependency = app.dependency_overrides[get_db]()
    db = next(dependency)
    try:
        asyncio.run(ingest_sample_library(db))
        assert len(db.scalars(select(Document)).all()) == 2
        owner = db.get(User, SAMPLE_LIBRARY_USER_ID)
        assert not owner.is_active
        chunks = RetrievalService()._chunks(db, account["user_id"])
        assert chunks and all(c.user_id == SAMPLE_LIBRARY_USER_ID for c in chunks)
        retrieval = RetrievalService()
        semantic = retrieval.semantic_search(db, account["user_id"], "Orion press warranty", 4)
        assert semantic and all(c.source == "semantic_chroma" for c in semantic)
        hits = ChromaVectorStore(retrieval.embedding_model).query(
            retrieval.embedding_model.embed_query("Orion press warranty"), account["user_id"], 4)
        assert hits and {hit.chunk_id for hit in hits} <= {c.chunk_id for c in chunks}
        assert retrieval.overview_search(db, account["user_id"], "Summarize the document")
    finally:
        dependency.close()
    assert client.get("/documents", headers=headers).json() == before


@pytest.mark.parametrize("client", [{"empty.txt": "   ", "valid.md": "The shared manual covers press maintenance."}], indirect=True)
def test_failed_sample_does_not_hide_other_samples_or_allow_querying_failed_file(client):
    token = signup(client)["access_token"]
    headers = auth_headers(token)
    docs = client.get("/documents", headers=headers).json()
    failed = next(d for d in docs if d["filename"] == "empty.txt")
    assert failed["status"] == "failed" and failed["error_message"]
    assert next(d for d in docs if d["filename"] == "valid.md")["status"] == "indexed"
    assert client.post("/chat/query", headers=headers, json={
        "question": "What is this about?", "document_ids": [failed["document_id"]]}).status_code == 409
