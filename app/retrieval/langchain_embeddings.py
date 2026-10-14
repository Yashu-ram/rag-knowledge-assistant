import math

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.errors import RAGServiceError


class EmbeddingModel(Embeddings):
    dimensions = 384

    def __init__(self):
        self.settings = get_settings()
        self._wrapper = None

    def _load_model(self):
        if self._wrapper is None:
            try:
                self._wrapper = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                raise RAGServiceError("EMBEDDING_UNAVAILABLE", "Could not load sentence-transformers/all-MiniLM-L6-v2. Check the model download and local dependencies.") from exc
        return self._wrapper._client

    def embed_documents(self, texts):
        self._load_model()
        try:
            vectors = self._wrapper.embed_documents(texts) if texts else []
            if len(vectors) != len(texts) or any(len(v) != self.dimensions or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("Expected finite 384-dimensional vectors")
            return vectors
        except Exception as exc:
            raise RAGServiceError("EMBEDDING_FAILED", "Embedding generation failed; no fallback embeddings were used.") from exc

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed(self, texts):
        return self.embed_documents(texts)


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise RAGServiceError("EMBEDDING_DIMENSION_MISMATCH", "Stored vectors do not match the embedding model. Reindex the affected documents.")
    return sum(x * y for x, y in zip(a, b))
