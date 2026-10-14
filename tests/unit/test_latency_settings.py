from unittest.mock import MagicMock

from app.generation import model_gateway
from app.retrieval import service


def test_retrieval_reuses_embedding_model(monkeypatch):
    service.shared_embedding_model.cache_clear()
    factory = MagicMock()
    monkeypatch.setattr(service, "EmbeddingModel", factory)
    monkeypatch.setattr(service, "ChromaVectorStore", MagicMock())
    try:
        first = service.RetrievalService()
        second = service.RetrievalService()
        assert first.embedding_model is second.embedding_model
        factory.assert_called_once()
    finally:
        service.shared_embedding_model.cache_clear()


def test_quick_generation_requests_short_answer_and_keeps_model_warm(monkeypatch):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {"claims": []}
    constructor = MagicMock(return_value=llm)
    monkeypatch.setattr(model_gateway, "ChatOllama", constructor)
    monkeypatch.setattr(model_gateway, "ensure_ollama_model", lambda value: value)
    text, model, _ = model_gateway.generate_answer("Question", [], "qwen2.5:3b", "quick")
    assert text == '{"claims": []}'
    assert model == "qwen2.5:3b"
    request = constructor.call_args.kwargs
    assert "at most 2 points" in llm.with_structured_output.return_value.invoke.call_args.args[0][1].content
    assert request["num_predict"] == 400
    assert request["keep_alive"] == "10m"
