"""HTTP transport shared by the UI and demo uploader; no RAG logic."""

import requests


def request_backend(method, base_url, path, *, token=None, timeout=None, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = timeout if timeout is not None else (300 if path == "/documents/upload" else 120)
    try:
        return requests.request(method, f"{base_url.rstrip('/')}{path}",
                                headers=headers, timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to the backend. Start FastAPI on port 8000 and check API_BASE_URL."
        ) from exc


def indexed_document_ids(documents):
    return {item["document_id"] for item in documents if item.get("status") == "indexed"}
