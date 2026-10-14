import pytest

from tests.conftest import auth_headers, signup
from tests.integration.test_rag_api import upload_text


@pytest.mark.parametrize("filename", ["visible.txt", "visible.md"])
def test_document_metadata_list_refresh_and_query(client, filename):
    token = signup(client)["access_token"]
    headers = auth_headers(token)
    assert client.get("/health").status_code == 200
    assert client.get("/documents", headers=headers).json() == []
    document = upload_text(client, token, filename, "The Orion press warranty lasts 31 months.")
    doc_id = document["document_id"]
    assert document["status"] == "indexed"
    assert document["analysis"]["chunk_count"] > 0
    for _ in range(2):
        listed = client.get("/documents", headers=headers).json()
        assert listed[0]["document_id"] == doc_id
        assert listed[0]["filename"] == filename
        assert listed[0]["created_at"]
        assert listed[0]["file_type"] in {".txt", ".md"}
        assert client.get(f"/documents/{doc_id}", headers=headers).json() == listed[0]
    result = client.post("/chat/query", headers=headers,
                         json={"question": "What is the Orion press warranty period?", "document_ids": [doc_id]})
    assert result.status_code == 200, result.text
    assert result.json()["citations"]
    assert all(c["document_id"] == doc_id for c in result.json()["citations"])


def test_failed_ingestion_is_visible_and_not_queryable(client):
    token = signup(client)["access_token"]
    headers = auth_headers(token)
    result = client.post("/documents/upload", headers=headers,
                         files={"file": ("blank.txt", b"   ", "text/plain")})
    assert result.status_code == 400
    docs = client.get("/documents", headers=headers).json()
    assert len(docs) == 1
    assert docs[0]["status"] == "failed"
    assert docs[0]["error_message"]
    result = client.post("/chat/query", headers=headers,
                         json={"question": "What is this document about?", "document_ids": [docs[0]["document_id"]]})
    assert result.status_code == 409


def test_document_list_is_account_scoped(client):
    first = signup(client, email="first@example.com")["access_token"]
    second = signup(client, email="second@example.com")["access_token"]
    upload_text(client, first, "private.txt", "Private document text.")
    assert client.get("/documents", headers=auth_headers(second)).json() == []
    assert client.get("/documents").status_code == 401
