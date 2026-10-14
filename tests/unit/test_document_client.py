from unittest.mock import Mock

import pytest
import requests

from app.ui.api_client import request_backend, indexed_document_ids
from scripts.ingest_samples import find_sample_files, upload_samples


@pytest.mark.parametrize("method,path", [("GET", "/documents"), ("POST", "/documents/upload"),
                                         ("POST", "/chat/query")])
def test_client_sends_auth_header(monkeypatch, method, path):
    request = Mock()
    monkeypatch.setattr(requests, "request", request)
    request_backend(method, "http://localhost:8000/", path, token="test-token")
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert request.call_args.args[1] == "http://localhost:8000" + path
    assert request.call_args.kwargs["timeout"] == (300 if path.endswith("upload") else 120)


def test_connection_failure_is_friendly(monkeypatch):
    monkeypatch.setattr(requests, "request", Mock(side_effect=requests.ConnectionError("raw socket detail")))
    with pytest.raises(RuntimeError, match="Cannot connect to the backend"):
        request_backend("GET", "http://localhost:8000", "/documents", token="secret")


def test_only_indexed_documents_are_selectable():
    assert indexed_document_ids([{"document_id": "ready", "status": "indexed"},
                                 {"document_id": "failed", "status": "failed"},
                                 {"document_id": "pending", "status": "processing"}]) == {"ready"}


def test_sample_discovery(tmp_path):
    for name in ["one.PDF", "two.txt", "three.md", "ignored.exe"]:
        (tmp_path / name).touch()
    nested = tmp_path / "folder"
    nested.mkdir()
    (nested / "four.txt").touch()
    assert {p.name for p in find_sample_files(tmp_path)} == {"one.PDF", "two.txt", "three.md", "four.txt"}


def test_sample_upload_uses_api_and_reports_metadata(tmp_path, monkeypatch, capsys):
    from scripts import ingest_samples
    (tmp_path / "sample.txt").write_text("Test content", encoding="utf-8")
    response = Mock(status_code=200)
    response.json.return_value = {"filename": "sample.txt", "document_id": "doc-1",
                                  "status": "indexed", "analysis": {"chunk_count": 2}}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    request = Mock(return_value=response)
    monkeypatch.setattr(ingest_samples, "request_backend", request)
    result = upload_samples("http://localhost:8000", "token", tmp_path)
    assert request.call_args.kwargs["token"] == "token"
    assert request.call_args.args[2] == "/documents/upload"
    assert result[0]["document_id"] == "doc-1"
    assert "sample.txt\tindexed\tdoc-1\tchunks=2" in capsys.readouterr().out
