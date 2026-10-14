from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from app.api.routes import auth, chat, conversations, documents, health, models
from app.storage.database import Base, engine
from app.core.config import get_settings
from app.retrieval.service import shared_embedding_model
from app.generation.model_gateway import list_ollama_models, warm_ollama_model
from app.core.errors import RAGServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    try:
        model = shared_embedding_model(settings.embedding_model)
        await run_in_threadpool(model.embed, ["Warm up document retrieval."])
        app.state.embedding_error = None
    except RAGServiceError as exc:
        app.state.embedding_error = exc.code
        logging.getLogger(__name__).warning("RAG startup: %s", exc.code)
    models = await run_in_threadpool(list_ollama_models)
    app.state.llm_warmed = False
    if settings.default_llm_model in models.get("installed_models", []):
        app.state.llm_warmed = await run_in_threadpool(warm_ollama_model, settings.default_llm_model)
    yield


def create_app() -> FastAPI:
    Base.metadata.create_all(bind=engine)
    app = FastAPI(title="HireKiwi RAG Knowledge Assistant", lifespan=lifespan)
    @app.exception_handler(RAGServiceError)
    async def service_error(request, exc):
        logging.getLogger(__name__).warning("%s %s: %s", request.method, request.url.path, exc.code)
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc), "error": {"code": exc.code, "message": str(exc)}})
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(models.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    return app


app = create_app()

