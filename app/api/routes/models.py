from fastapi import APIRouter

from app.generation.model_gateway import list_ollama_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def models() -> dict:
    return list_ollama_models()

