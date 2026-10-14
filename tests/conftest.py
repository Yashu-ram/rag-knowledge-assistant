import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.main import app
from app.storage.database import Base, get_db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.api.routes import chat as chat_routes
    from app.retrieval import service, vector_store
    from tests.fakes import TestEmbeddings, grounded_generator
    from app import main as main_module

    fake = TestEmbeddings()
    service.shared_embedding_model.cache_clear()
    monkeypatch.setattr(service, "EmbeddingModel", lambda: fake)
    monkeypatch.setattr(vector_store, "EmbeddingModel", lambda: fake)
    monkeypatch.setattr(chat_routes, "ensure_ollama_model", lambda model=None: model or "qwen2.5:1.5b")
    monkeypatch.setattr(main_module, "list_ollama_models", lambda: {"available": False, "installed_models": []})

    monkeypatch.setattr(chat_routes, "generate_answer", grounded_generator)
    settings = get_settings()
    settings.upload_dir = tmp_path / "uploads"
    settings.chroma_dir = tmp_path / "chroma"
    settings.jwt_secret_key = "test-secret-key-with-at-least-32-bytes"
    settings.jwt_expire_minutes = 60
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    service.shared_embedding_model.cache_clear()
    # Release per-test HNSW workers and Windows file mappings, not production data.
    from chromadb.api.client import SharedSystemClient
    for system in list(SharedSystemClient._identifier_to_system.values()):
        system.stop()
    SharedSystemClient.clear_system_cache()
    engine.dispose()


def signup(client: TestClient, email: str = "person@example.com", password: str = "strong-password-123") -> dict:
    response = client.post(
        "/auth/signup",
        json={"full_name": "Test Person", "email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

