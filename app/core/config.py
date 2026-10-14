from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "HireKiwi RAG Knowledge Assistant"
    database_url: str = "sqlite:///./data/app.db"
    upload_dir: Path = Path("./uploads")
    chroma_dir: Path = Path("./chroma_store")
    max_upload_mb: int = 20
    pdf_extraction_timeout_seconds: int = Field(default=180, ge=1, le=240)
    jwt_secret_key: str = Field(default="change-this-local-secret")
    jwt_expire_minutes: int = 120
    embedding_model: Literal["sentence-transformers/all-MiniLM-L6-v2"] = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection: str = "document_chunks"
    ollama_base_url: str = "http://localhost:11434"
    default_top_k: int = 6
    debug_llm_output: bool = False
    chunk_target_tokens: int = Field(default=240, ge=64, le=248)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=128)
    max_document_chars: int = Field(default=5_000_000, ge=1000)
    max_generation_passages: int = Field(default=12, ge=1, le=32)
    max_generation_context_chars: int = Field(default=8000, ge=2000, le=24000)
    default_llm_model: Literal["qwen2.5:1.5b", "qwen2.5:3b"] = "qwen2.5:1.5b"

    @property
    def allowed_llm_models(self) -> tuple[str, str]:
        return ("qwen2.5:1.5b", "qwen2.5:3b")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    return settings

