import json
from pathlib import Path

import pytest

from app.api.routes import chat
from tests.conftest import auth_headers, signup
from tests.integration.test_rag_api import upload_text


@pytest.mark.parametrize("separate", [False, True])
def test_comparison_api_cites_each_requested_fact(client, monkeypatch, separate):
    token = signup(client)["access_token"]
    texts = ["Alpha shipping takes 5 days.", "Beta shipping takes 9 days."]
    docs = [upload_text(client, token, "alpha.txt", texts[0]), upload_text(client, token, "beta.txt", texts[1])] if separate else [upload_text(client, token, "shipping.txt", " ".join(texts))]
    def generate(question, evidence, model, mode, feedback=None):
        claims = [{"text": text, "source_id": next(i for i, item in enumerate(evidence, 1) if text in item.text)} for text in texts]
        return json.dumps({"claims": claims}), "test-model", 1
    monkeypatch.setattr(chat, "generate_answer", generate)
    result = client.post("/chat/query", headers=auth_headers(token), json={
        "question": "Compare Alpha shipping and Beta shipping.", "document_ids": [d["document_id"] for d in docs]})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["answerable"] == "answerable"
    assert "5 days" in data["answer"] and "9 days" in data["answer"]
    assert len(data["citations"]) == 2
    assert len(data["retrieval_metadata"]["completeness"]["covered_parts"]) == 2


@pytest.mark.parametrize("question,texts", [
    ("Compare fresh loan WALR and outstanding loan WALR in February 2026.", [
        "Fresh rupee loan WALR increased by 5 basis points.", "Outstanding rupee loan WALR declined by 26 basis points."]),
    ("How did the median 1-year MCLR change in H2:2025-26?", [
        "The median one-year MCLR declined by 20 basis points in H2:2025-26."]),
])
def test_rbi_synthesis_regressions(client, monkeypatch, question, texts):
    pdf = Path(__file__).resolve().parents[2] / "sample_documents/RBI_Policy_RAG_Sample.pdf"
    token = signup(client)["access_token"]
    document = client.post("/documents/upload", headers=auth_headers(token), files={"file": (pdf.name, pdf.read_bytes())}).json()
    def generate(_, evidence, model, mode, feedback=None):
        from app.generation.grounded_answer import supporting_snippet, context_records
        records = context_records(evidence)
        claims = []
        for text in texts:
            record = max(records, key=lambda r: supporting_snippet(text, r)[1])
            claims.append({"text": text, "source_id": record["source_id"]})
        return json.dumps({"claims": claims}), "test-model", 1
    monkeypatch.setattr(chat, "generate_answer", generate)
    result = client.post("/chat/query", headers=auth_headers(token), json={"question": question, "document_ids": [document["document_id"]]})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["answerable"] == "answerable"
    for text in texts:
        assert text in data["answer"]
    assert all(c["source_filename"] == pdf.name for c in data["citations"])
    assert not data["retrieval_metadata"]["completeness"]["missing_supported_parts"]
