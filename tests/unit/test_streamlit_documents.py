from pathlib import Path
from unittest.mock import Mock
import json

import requests
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app/ui/streamlit_app.py"


def document(doc_id="doc-1", status="indexed"):
    return {"document_id": doc_id, "filename": f"{doc_id}.txt", "file_type": ".txt",
            "status": status, "created_at": "2026-10-14T12:00:00",
            "error_message": None, "analysis": {"chunk_count": 2}}


def setup_app(monkeypatch, documents):
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = Mock(status_code=200)
        if url.endswith("/documents"):
            payload = list(documents)
        elif url.endswith("/models"):
            payload = {"available": True, "installed_models": ["qwen2.5:1.5b"]}
        else:
            payload = []
        response.json.return_value = payload
        response.content = json.dumps(payload).encode()
        return response
    monkeypatch.setattr(requests, "request", request)
    app = AppTest.from_file(str(APP), default_timeout=15)
    app.session_state["token"] = "test-token"
    app.session_state["user"] = {"full_name": "Test Person", "email": "test@example.com"}
    return app, calls


def test_ui_lists_backend_documents_and_keeps_selection(monkeypatch):
    docs = [document(), document("failed", "failed"), document("pending", "processing")]
    app, calls = setup_app(monkeypatch, docs)
    app.run()
    assert not app.exception
    assert app.multiselect(key="document_selector").options == ["doc-1.txt (doc-1)"]
    assert any("completed / indexed" in item.label for item in app.expander)
    assert any("processing" in item.label for item in app.expander)
    assert any("failed" in item.label for item in app.expander)
    app.multiselect(key="document_selector").set_value(["doc-1"]).run()
    assert not app.exception
    assert app.session_state["selected_document_ids"] == ["doc-1"]
    app.run()
    assert app.session_state["selected_document_ids"] == ["doc-1"]
    assert all(call[2]["headers"]["Authorization"] == "Bearer test-token" for call in calls)


def test_new_user_can_select_shared_samples_without_uploading(monkeypatch):
    sample = dict(document("shared-demo"), is_sample=True)
    app, _ = setup_app(monkeypatch, [sample])
    app.run()
    assert not app.exception
    assert any("shared sample" in item.label for item in app.expander)
    assert app.multiselect(key="document_selector").options == ["shared-demo.txt [sample] (shared-d)"]
    app.multiselect(key="document_selector").set_value(["shared-demo"]).run()
    assert not app.exception
    assert app.session_state["selected_document_ids"] == ["shared-demo"]
    assert not any("No documents uploaded" in item.value for item in app.info)


def test_refresh_shows_externally_ingested_files_and_removes_deleted_selection(monkeypatch):
    docs = [document()]
    app, _ = setup_app(monkeypatch, docs)
    app.run()
    app.multiselect(key="document_selector").set_value(["doc-1"]).run()
    docs[:] = [document("doc-2")]
    app.button(key="refresh_documents").click().run()
    assert not app.exception
    assert app.multiselect(key="document_selector").options == ["doc-2.txt (doc-2)"]
    assert app.session_state["selected_document_ids"] == []


def test_empty_documents_show_upload_and_warning(monkeypatch):
    app, _ = setup_app(monkeypatch, [])
    app.run()
    assert not app.exception
    assert any("No documents uploaded" in item.value for item in app.info)
    assert any("Select an indexed document" in item.value for item in app.warning)
    assert any(item.label == "Upload selected files" for item in app.button)


def test_backend_failure_is_not_presented_as_empty_documents(monkeypatch):
    app, _ = setup_app(monkeypatch, [])
    monkeypatch.setattr(requests, "request", Mock(side_effect=requests.ConnectionError()))
    app.run()
    assert not app.exception
    assert any("Cannot connect to the backend" in item.value for item in app.error)
    assert not any("No documents uploaded" in item.value for item in app.info)


def test_login_stores_token_and_loads_authenticated_documents(monkeypatch):
    calls = []
    def request(method, url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/auth/login"):
            data = {"access_token": "login-token", "full_name": "Test Person", "email": "test@example.com"}
        elif url.endswith("/documents"):
            data = [document()]
        elif url.endswith("/models"):
            data = {"available": True, "installed_models": ["qwen2.5:1.5b"]}
        else:
            data = []
        response = Mock(status_code=200, content=b"json")
        response.json.return_value = data
        return response
    monkeypatch.setattr(requests, "request", request)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.text_input[0].set_value("test@example.com")
    app.text_input[1].set_value("test password")
    next(item for item in app.button if item.label == "Log in").click().run()
    assert not app.exception
    assert app.session_state["token"] == "login-token"
    doc_requests = [kwargs for url, kwargs in calls if url.endswith("/documents")]
    assert doc_requests
    assert doc_requests[0]["headers"]["Authorization"] == "Bearer login-token"
