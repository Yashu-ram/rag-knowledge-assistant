import json
import pytest
import pymupdf

from app.core.errors import RAGServiceError
from app.ingestion import langchain_documents
from app.main import app
from app.storage.database import get_db
from app.storage.models import Document
from tests.conftest import signup, auth_headers


def test_stream_reports_real_stages_then_indexed_result(client):
    headers = auth_headers(signup(client)["access_token"])
    response = client.post("/documents/upload?stream=true", headers=headers,
                           files={"file": ("progress.txt", b"The warranty period is 29 months.")})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e["stage"] for e in events if e["type"] == "progress"] == [
        "Uploading file", "Extracting text", "Chunking", "Creating embeddings", "Indexing in ChromaDB",
    ]
    assert events[-1]["type"] == "result"
    assert events[-1]["document"]["status"] == "indexed"


def test_blank_pdf_extraction_fails_clearly_and_cannot_be_queried(client):
    headers = auth_headers(signup(client)["access_token"])
    with pymupdf.open() as pdf:
        pdf.new_page()
        data = pdf.tobytes()
    response = client.post("/documents/upload", headers=headers, files={"file": ("blank.pdf", data)})
    assert response.status_code == 400
    assert "no extractable text" in response.json()["detail"].lower()
    document = client.get("/documents", headers=headers).json()[0]
    assert document["status"] == "failed"
    query = client.post("/chat/query", headers=headers,
                        json={"question": "What does it say?", "document_ids": [document["document_id"]]})
    assert query.status_code == 409


def test_pdf_timeout_preserves_clear_error_in_stream(client, monkeypatch):
    headers = auth_headers(signup(client)["access_token"])
    def timeout(*args):
        raise RAGServiceError("PDF_EXTRACTION_TIMEOUT", "PDF extraction exceeded 180 seconds.", 504)
    monkeypatch.setattr(langchain_documents, "load_pdf_with_timeout", timeout)
    response = client.post("/documents/upload?stream=true", headers=headers,
                           files={"file": ("slow.pdf", b"PDF bytes")})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "PDF_EXTRACTION_TIMEOUT"
    assert events[-1]["status_code"] == 504
    assert not any(e["type"] == "result" for e in events)
    assert client.get("/documents", headers=headers).json()[0]["status"] == "failed"


def test_scanned_pdf_without_ocr_has_requested_message(client, monkeypatch):
    message = "No extractable text found. This PDF may be scanned and OCR is not enabled."
    def fail(*args):
        raise ValueError(message)
    monkeypatch.setattr(langchain_documents, "load_pdf_with_timeout", fail)
    headers = auth_headers(signup(client)["access_token"])
    response = client.post("/documents/upload", headers=headers, files={"file": ("scan.pdf", b"PDF bytes")})
    assert response.status_code == 400 and message in response.json()["detail"]


@pytest.mark.parametrize("explicit", [True, False])
def test_processing_document_blocks_queries(client, explicit):
    headers = auth_headers(signup(client)["access_token"])
    document = client.post("/documents/upload", headers=headers,
                           files={"file": ("guide.txt", b"The warranty lasts 29 months.")}).json()
    provider = app.dependency_overrides[get_db]()
    db = next(provider)
    db.get(Document, document["document_id"]).status = "processing"
    db.commit()
    provider.close()
    payload = {"question": "What is the warranty period?"}
    if explicit:
        payload["document_ids"] = [document["document_id"]]
    assert client.post("/chat/query", headers=headers, json=payload).status_code == 409
