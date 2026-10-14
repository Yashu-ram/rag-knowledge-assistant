from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import TextSplitter
from app.core.errors import RAGServiceError
from app.generation import model_gateway
from app.retrieval import langchain_embeddings
from app.ingestion.chunker import SentenceAwareSplitter
from app.ingestion.langchain_documents import StructuredDocumentLoader


def test_embedding_load_failure_never_returns_fake_vectors(monkeypatch):
    monkeypatch.setattr(langchain_embeddings, "HuggingFaceEmbeddings", MagicMock(side_effect=RuntimeError("offline")))
    with pytest.raises(RAGServiceError, match="Could not load") as error:
        langchain_embeddings.EmbeddingModel().embed(["text"])
    assert error.value.code == "EMBEDDING_UNAVAILABLE"


def test_wrong_embedding_dimension_fails(monkeypatch):
    wrapper = MagicMock()
    wrapper.embed_documents.return_value = [[1.0] * 128]
    monkeypatch.setattr(langchain_embeddings, "HuggingFaceEmbeddings", lambda **kwargs: wrapper)
    with pytest.raises(RAGServiceError) as error:
        langchain_embeddings.EmbeddingModel().embed(["text"])
    assert error.value.code == "EMBEDDING_FAILED"


def test_langchain_loader_and_splitter_preserve_metadata(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Warranty\nThe warranty lasts 24 months.", encoding="utf-8")
    loader = StructuredDocumentLoader(path, "d1", "guide.md", ".md")
    assert isinstance(loader, BaseLoader)
    splitter = SentenceAwareSplitter(100, 20, lambda text: len(text.split()))
    assert isinstance(splitter, TextSplitter)
    chunks = splitter.split_documents(loader.load())
    assert chunks and all(isinstance(d, Document) for d in chunks)
    assert chunks[0].metadata["document_id"] == "d1"
    assert chunks[0].metadata["source"] == "guide.md"
    assert chunks[0].metadata["chunk_id"]


def test_model_list_hides_other_installed_models(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"models": [{"name": "qwen2.5:3b"}, {"name": "llama3:8b"}]}
    monkeypatch.setattr(model_gateway.httpx, "get", lambda *args, **kwargs: response)
    status = model_gateway.list_ollama_models()
    assert status["models"] == ["qwen2.5:1.5b", "qwen2.5:3b"]
    assert status["installed_models"] == ["qwen2.5:3b"]
    assert status["missing_models"] == ["qwen2.5:1.5b"]


@pytest.mark.parametrize("available,installed,code", [
    (False, [], "OLLAMA_UNAVAILABLE"), (True, [], "MODEL_NOT_INSTALLED"),
])
def test_runtime_failures_are_not_abstentions(monkeypatch, available, installed, code):
    monkeypatch.setattr(model_gateway, "list_ollama_models", lambda: {
        "available": available, "installed_models": installed, "error": "Ollama unavailable",
    })
    with pytest.raises(RAGServiceError) as error:
        model_gateway.generate_answer("question", [], "qwen2.5:3b", "quick")
    assert error.value.code == code


def test_generation_failure_is_clear(monkeypatch):
    monkeypatch.setattr(model_gateway, "ensure_ollama_model", lambda model: model)
    monkeypatch.setattr(model_gateway, "ChatOllama", MagicMock(side_effect=RuntimeError("broken")))
    with pytest.raises(RAGServiceError) as error:
        model_gateway.generate_answer("question", [], "qwen2.5:3b", "quick")
    assert error.value.code == "LLM_GENERATION_FAILED"
