import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from app.core.errors import RAGServiceError
from app.ingestion import pdf_timeout
from app.ui.api_client import request_backend


def ui_functions():
    path = Path(__file__).resolve().parents[2] / "app/ui/streamlit_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"api", "upload_files", "submit_question"}]
    st = MagicMock()
    st.session_state = {"chat_documents": {}, "messages": [], "conversation_documents": {},
                        "selected_document_ids": [], "upload_errors": []}
    namespace = {"st": st, "requests": requests, "json": json, "Any": Any, "API_BASE_URL": "http://testserver",
                 "request_backend": request_backend, "logger": MagicMock(), "refresh_documents": MagicMock()}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, st


@pytest.mark.parametrize("path,timeout", [("/documents/upload", 300), ("/health", 120)])
def test_upload_client_has_300_second_timeout(monkeypatch, path, timeout):
    functions, _ = ui_functions()
    response = MagicMock(status_code=200, content=b"{}")
    request = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "request", request)
    functions["api"]("POST", path)
    assert request.call_args.kwargs["timeout"] == timeout


def test_failed_upload_never_sends_question(monkeypatch):
    functions, st = ui_functions()
    request = MagicMock(side_effect=requests.Timeout("slow"))
    monkeypatch.setattr(requests, "request", request)
    file = SimpleNamespace(name="slow.pdf", getvalue=lambda: b"PDF bytes")
    functions["submit_question"]("Question", [file], None, "quick", None)
    assert request.call_count == 1
    assert not st.session_state["chat_documents"]
    assert st.session_state["messages"] == []
    assert "300 seconds" in st.error.call_args.args[0]


def test_ui_consumes_progress_before_accepting_indexed_result(monkeypatch):
    functions, st = ui_functions()
    stages = ["Uploading file", "Extracting text", "Chunking", "Creating embeddings", "Indexing in ChromaDB"]
    response = MagicMock(status_code=200)
    result = {"document_id": "d1", "filename": "guide.pdf", "status": "indexed"}
    response.iter_lines.return_value = [json.dumps({"type": "progress", "stage": stage}).encode() for stage in stages] + [
        json.dumps({"type": "result", "document": result}).encode()]
    request = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "request", request)
    file = SimpleNamespace(name="guide.pdf", getvalue=lambda: b"PDF bytes")
    assert functions["upload_files"]([file]) == ["guide.pdf"]
    assert request.call_args.kwargs["timeout"] == 300
    assert request.call_args.kwargs["params"]["stream"] == "true"
    response.iter_lines.assert_called_once_with(chunk_size=1)
    assert [call.args[0] for call in st.status.return_value.write.call_args_list] == stages
    assert st.session_state["chat_documents"] == {"d1": "guide.pdf"}
    functions["refresh_documents"].assert_called_once()
    assert st.session_state["selected_document_ids"] == ["d1"]


def test_stream_error_is_not_accepted_as_upload_success(monkeypatch):
    functions, st = ui_functions()
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [b'{"type":"error","message":"No extractable text found."}']
    monkeypatch.setattr(requests, "request", MagicMock(return_value=response))
    file = SimpleNamespace(name="empty.pdf", getvalue=lambda: b"PDF bytes")
    assert functions["upload_files"]([file]) == []
    assert not st.session_state["chat_documents"]
    assert st.session_state["upload_failed"]


def test_extraction_timeout_terminates_worker(monkeypatch):
    run = MagicMock(side_effect=pdf_timeout.subprocess.TimeoutExpired("worker", 180))
    monkeypatch.setattr(pdf_timeout.subprocess, "run", run)
    with pytest.raises(RAGServiceError) as error:
        pdf_timeout.load_pdf_with_timeout("slow.pdf", 180)
    assert error.value.code == "PDF_EXTRACTION_TIMEOUT"
    assert run.call_args.kwargs["timeout"] == 180
    assert run.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
