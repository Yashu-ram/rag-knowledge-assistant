import pymupdf

from tests.conftest import signup, auth_headers


def test_text_based_pdf_ingests_and_answers_with_page_citation(client):
    headers = auth_headers(signup(client)["access_token"])
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((45, 65), "The Orion press warranty lasts 31 months.")
        content = pdf.tobytes()
    upload = client.post("/documents/upload", headers=headers, files={"file": ("orion.pdf", content)})
    assert upload.status_code == 200, upload.text
    doc = upload.json()
    assert doc["status"] == "indexed" and doc["analysis"]["chunk_count"] >= 1
    response = client.post("/chat/query", headers=headers, json={
        "question": "What is the Orion press warranty?", "document_ids": [doc["document_id"]],
    })
    assert response.status_code == 200, response.text
    assert "31 months" in response.json()["answer"]
    citation = response.json()["citations"][0]
    assert citation["page_number"] == 1 and citation["source_filename"] == "orion.pdf"
    assert citation["chunk_id"] and "31 months" in citation["excerpt"]


def test_pipeline_uses_eight_candidates_per_search_and_at_most_six_contexts(client):
    headers = auth_headers(signup(client)["access_token"])
    content = "\n\n".join(f"Maintenance procedure {i} requires filter inspection before use. " * 35 for i in range(12))
    doc = client.post("/documents/upload", headers=headers,
                      files={"file": ("maintenance.txt", content.encode())}).json()
    response = client.post("/chat/query", headers=headers, json={
        "question": "What inspection does the maintenance procedure require?", "document_ids": [doc["document_id"]],
    })
    assert response.status_code == 200, response.text
    metadata = response.json()["retrieval_metadata"]
    diagnostics = metadata["tool_calls"][0]["diagnostics"]
    assert len(diagnostics["semantic_chunk_ids"]) == 8
    assert len(diagnostics["keyword_chunk_ids"]) == 8
    assert 4 <= len(diagnostics["final_chunk_ids"]) <= 6
