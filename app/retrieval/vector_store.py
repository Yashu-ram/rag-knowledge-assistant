from dataclasses import dataclass
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.storage.models import Chunk
from app.storage.document_access import readable_owner_ids
from app.retrieval.embeddings import EmbeddingModel
from app.core.errors import RAGServiceError


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


class ChromaVectorStore:
    def __init__(self, embedding_model=None) -> None:
        settings = get_settings()
        self._available = False
        self._collection = None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(settings.chroma_dir),
                settings=chromadb.config.Settings(anonymized_telemetry=False))
            self.store = Chroma(client=client, collection_name=settings.chroma_collection,
                                embedding_function=embedding_model or EmbeddingModel(),
                                collection_metadata={"hnsw:space": "cosine"})
            self._collection = self.store._collection
            existing = self._collection.get(limit=1, include=["embeddings"])["embeddings"]
            if existing is not None and len(existing) and len(existing[0]) != 384:
                raise RAGServiceError("EMBEDDING_DIMENSION_MISMATCH", "Chroma contains incompatible vectors. Use a new collection and reindex; existing uploads are preserved.")
            self._available = True
        except RAGServiceError:
            raise
        except Exception as exc:
            raise RAGServiceError("VECTOR_STORE_UNAVAILABLE", "Could not open the persistent Chroma index. Check permissions and database compatibility.") from exc

    @property
    def available(self) -> bool:
        return self._available and self._collection is not None

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not self.available or not chunks:
            return
        if any(len(vector) != 384 for vector in embeddings):
            raise RAGServiceError("EMBEDDING_DIMENSION_MISMATCH", "Expected 384-dimensional embeddings for Chroma.")
        try:
            self._collection.upsert(ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks], metadatas=[self._metadata(c) for c in chunks],
                embeddings=embeddings)
        except RAGServiceError:
            raise
        except Exception as exc:
            raise RAGServiceError("VECTOR_INDEX_FAILED", "Could not store document vectors in Chroma.") from exc

    def search(self, query, user_id, top_k, document_ids=None):
        owners = readable_owner_ids(user_id)
        where = {"user_id": {"$in": owners}}
        if document_ids:
            where = {"$and": [where, {"document_id": {"$in": document_ids}}]}
        try:
            count = self._collection.count()
            if not count:
                return []
            retriever = self.store.as_retriever(search_type="similarity", search_kwargs={"k": min(top_k, count), "filter": where})
            documents = retriever.invoke(query)
            return [VectorHit(d.metadata.get("chunk_id") or d.id, 1.0 / (rank + 1)) for rank, d in enumerate(documents)
                    if d.metadata["user_id"] in owners and (not document_ids or d.metadata["document_id"] in document_ids)]
        except RAGServiceError:
            raise
        except Exception as exc:
            raise RAGServiceError("VECTOR_RETRIEVAL_FAILED", "Chroma retrieval failed. Check index/embedding compatibility and reindex if necessary.") from exc

    def query(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        if not self.available:
            return []
        owners = readable_owner_ids(user_id)
        where = {"user_id": {"$in": owners}}
        if document_ids:
            where = {"$and": [where, {"document_id": {"$in": document_ids}}]}
        count = self._collection.count()
        if not count:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=["distances", "metadatas"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        allowed_documents = set(document_ids or [])
        hits: list[VectorHit] = []
        for chunk_id, distance, metadata in zip(ids, distances, metadatas):
            if metadata.get("user_id") not in owners:
                continue
            if allowed_documents and metadata.get("document_id") not in allowed_documents:
                continue
            hits.append(VectorHit(chunk_id=chunk_id, score=1.0 / (1.0 + float(distance))))
            if len(hits) >= top_k:
                break
        return hits

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not self.available or not chunk_ids:
            return
        self.store.delete(ids=chunk_ids)

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.source_filename,
            "source": chunk.source_filename,
            "section": getattr(chunk, "source_section", ""),
            "user_id": chunk.user_id,
            "document_id": chunk.document_id,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number if chunk.page_number is not None else -1,
            "chunk_index": chunk.chunk_index,
        }

