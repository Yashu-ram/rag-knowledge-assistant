from tests.conftest import auth_headers, signup


def upload_text(client, token: str, filename: str, content: str) -> dict:
    response = client.post(
        "/documents/upload",
        headers=auth_headers(token),
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_chat_citations_and_delete(client):
    user = signup(client)
    token = user["access_token"]
    document = upload_text(
        client,
        token,
        "kiwidesk.txt",
        (
            "KiwiDesk 4.2 is the current supported release. "
            "Smart Inbox routes messages by urgency, account tier, and language. "
            "The KD-401 upload error means document indexing is not enabled."
        ),
    )
    assert document["status"] == "indexed"

    chat = client.post(
        "/chat",
        headers=auth_headers(token),
        json={"question": "What does Smart Inbox route by?", "document_ids": [document["document_id"]]},
    )
    assert chat.status_code == 200, chat.text
    data = chat.json()
    assert data["answerable"] in {"answerable", "partial"}
    assert data["citations"]
    assert data["citations"][0]["source_filename"] == "kiwidesk.txt"
    assert "Smart Inbox" in data["citations"][0]["excerpt"]
    assert data["retrieval_metadata"]["tool_calls"]
    assert data["retrieval_metadata"]["grounding"]["faithfulness"] == 1.0
    for citation in data["citations"]:
        assert f"[{citation['citation_id']}](#{citation['anchor']})" in data["answer"]

    deleted = client.delete(f"/documents/{document['document_id']}", headers=auth_headers(token))
    assert deleted.status_code == 200

    after_delete = client.post(
        "/chat",
        headers=auth_headers(token),
        json={"question": "What does Smart Inbox route by?", "document_ids": []},
    )
    assert after_delete.status_code == 200
    assert after_delete.json()["answerable"] == "unanswerable"


def test_user_cannot_access_other_users_documents(client):
    user_a = signup(client, email="a@example.com")
    user_b = signup(client, email="b@example.com")
    document = upload_text(
        client,
        user_a["access_token"],
        "private.txt",
        "Private answer: KiwiDesk internal code is BLUE-77.",
    )

    get_attempt = client.get(
        f"/documents/{document['document_id']}",
        headers=auth_headers(user_b["access_token"]),
    )
    assert get_attempt.status_code == 404

    scoped_chat = client.post(
        "/chat",
        headers=auth_headers(user_b["access_token"]),
        json={"question": "What is the internal code?", "document_ids": [document["document_id"]]},
    )
    assert scoped_chat.status_code == 404


def test_unanswerable_empty_index(client):
    user = signup(client)
    response = client.post(
        "/chat",
        headers=auth_headers(user["access_token"]),
        json={"question": "What is the retention policy?"},
    )
    assert response.status_code == 200
    assert response.json()["answerable"] == "unanswerable"


def test_missing_personal_fact_can_reach_llm_but_has_no_citations(client, monkeypatch):
    from app.api.routes import chat as chat_routes
    from app.generation.grounding import ABSTENTION
    from unittest.mock import Mock

    user = signup(client)
    document = upload_text(client, user["access_token"], "api.txt",
                           "The filename field stores the name of the uploaded document. What is my name?")
    generator = Mock(return_value=('{"claims": []}', "test-model", 1))
    monkeypatch.setattr(chat_routes, "generate_answer", generator)
    response = client.post("/chat", headers=auth_headers(user["access_token"]),
                           json={"question": "what is my name", "document_ids": [document["document_id"]]})
    assert response.status_code == 200
    assert response.json()["answer"] == ABSTENTION
    assert response.json()["citations"] == []
    assert response.json()["retrieval_metadata"]["model_latency_ms"] == 1
    generator.assert_called_once()


def test_invalid_model_source_is_never_displayed(client, monkeypatch):
    from app.api.routes import chat as chat_routes
    from app.generation.grounding import ABSTENTION

    user = signup(client)
    document = upload_text(client, user["access_token"], "api.txt",
                           "For a 401 error, verify the bearer token.")
    monkeypatch.setattr(chat_routes, "generate_answer",
                        lambda *args: ('{"claims": [{"source_id": 999, "text": "Invented", "quote": "Invented"}]}', "test-model", 1))
    response = client.post("/chat", headers=auth_headers(user["access_token"]),
                           json={"question": "How do I fix a 401 error?", "document_ids": [document["document_id"]]})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == ABSTENTION
    assert data["citations"] == []
    assert data["retrieval_metadata"]["grounding"]["rejected_claims"] == 1


def test_selected_document_vector_filter_and_hybrid_search(client):
    user = signup(client)
    token = user["access_token"]
    target = upload_text(client, token, "target.txt", "The ZK-999 error requires restarting the worker.")
    upload_text(client, token, "distractor.txt", "The PDF file size limit is twenty megabytes.")
    response = client.post("/chat", headers=auth_headers(token),
                           json={"question": "What does the ZK-999 error require?",
                                 "document_ids": [target["document_id"]]})
    assert response.status_code == 200
    data = response.json()
    assert data["citations"]
    assert all(row["source_filename"] == "target.txt" for row in data["retrieval_metadata"]["retrieved_chunks"])
    diagnostics = data["retrieval_metadata"]["tool_calls"][0]["diagnostics"]
    assert diagnostics["semantic_chunk_ids"]
    assert diagnostics["keyword_chunk_ids"]
    assert any(row["search_source"] == "keyword+semantic_chroma"
               for row in data["retrieval_metadata"]["retrieved_chunks"])

