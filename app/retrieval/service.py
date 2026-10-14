import json
import time
import logging
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.retrieval.embeddings import EmbeddingModel, cosine_similarity
from app.retrieval.fusion import RetrievedChunk, reciprocal_rank_fusion
from app.retrieval.keyword_search import bm25_scores
from app.retrieval.vector_store import ChromaVectorStore
from app.storage.models import Chunk, Document, DocumentPage
from app.core.config import get_settings
from app.ingestion.profile import INDEX_VERSION

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def shared_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel()


class RetrievalService:
    def __init__(self) -> None:
        settings = get_settings()
        self.embedding_model = shared_embedding_model(settings.embedding_model)
        self.vector_store = ChromaVectorStore(self.embedding_model)
        self.search_diagnostics = {}

    def _chunks(self, db: Session, user_id: str, document_ids: list[str] | None = None) -> list[Chunk]:
        stmt = select(Chunk).join(Document, Chunk.document_id == Document.document_id).where(Chunk.user_id == user_id, Document.status == "indexed")
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return list(db.scalars(stmt).all())

    def semantic_search(self, db: Session, user_id: str, query: str, top_k: int, document_ids: list[str] | None = None, *, chunks=None) -> list[RetrievedChunk]:
        chunks = self._chunks(db, user_id, document_ids) if chunks is None else chunks
        if not chunks:
            return []
        chroma_hits = self.vector_store.search(query, user_id, top_k, document_ids)
        if chroma_hits:
            query_vector = self.embedding_model.embed_query(query)
            chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            results = []
            for hit in chroma_hits:
                chunk = chunks_by_id.get(hit.chunk_id)
                if chunk is not None:
                    score = cosine_similarity(query_vector, json.loads(chunk.embedding_json))
                    item = self._to_retrieved(chunk, score, "semantic_chroma")
                    item.metadata["semantic_score"] = score
                    results.append(item)
            if results:
                return results

        query_vector = self.embedding_model.embed_query(query)
        scored = []
        for chunk in chunks:
            if not chunk.embedding_json:
                continue
            vector = json.loads(chunk.embedding_json)
            scored.append((cosine_similarity(query_vector, vector), chunk))
        return [self._to_retrieved(chunk, score, "semantic") for score, chunk in sorted(scored, reverse=True, key=lambda item: item[0])[:top_k]]

    def retrieve(self, db, user_id, query, document_ids=None, *, overview=False, final_k=6):
        scope = document_ids
        if overview:
            selection = select(Document).where(Document.user_id == user_id, Document.status == "indexed")
            if scope:
                selection = selection.where(Document.document_id.in_(scope))
            document = db.scalar(selection.order_by(Document.created_at.desc(), Document.document_id).limit(1))
            if document is None:
                self.search_diagnostics = {}
                return []
            scope = [document.document_id]
        chunks = self._chunks(db, user_id, scope)
        semantic = self.semantic_search(db, user_id, query, 8, scope, chunks=chunks)
        keyword = self.keyword_search(db, user_id, query, 8, scope, chunks=chunks)
        logger.info("vector results count=%d", len(semantic))
        logger.info("BM25 results count=%d", len(keyword))
        fused = reciprocal_rank_fusion([semantic, keyword])
        logger.info("RRF fused results count=%d", len(fused))
        from app.generation.grounding import terms
        query_terms = terms(query)
        def rerank(item):
            coverage = len(query_terms & terms(item.text)) / max(1, len(query_terms))
            return item.score + 0.1 * coverage + 0.05 * item.metadata.get("semantic_score", 0)
        ranked = sorted(fused, key=rerank, reverse=True)
        starts = [self._to_retrieved(c, 0, "document_start")
                  for c in sorted(chunks, key=lambda c: c.chunk_index)[:2]] if overview else []
        unique = {}
        for item in starts + ranked:
            unique.setdefault(item.chunk_id, item)
        final = list(unique.values())[:final_k]
        final = self._expand_units(db, final, {c.chunk_id: c for c in chunks})
        self.search_diagnostics = {
            "semantic_chunk_ids": [i.chunk_id for i in semantic],
            "keyword_chunk_ids": [i.chunk_id for i in keyword],
            "fused_chunk_ids": [i.chunk_id for i in fused],
            "first_chunk_ids": [i.chunk_id for i in starts],
            "final_chunk_ids": [i.chunk_id for i in final],
        }
        logger.info("reranked final chunks count=%d chunk_ids=%s", len(final), self.search_diagnostics["final_chunk_ids"])
        return final

    def keyword_search(self, db: Session, user_id: str, query: str, top_k: int, document_ids: list[str] | None = None, *, chunks=None) -> list[RetrievedChunk]:
        chunks = self._chunks(db, user_id, document_ids) if chunks is None else chunks
        if not chunks:
            return []
        scores = bm25_scores(query, [chunk.text for chunk in chunks])
        ranked = sorted(zip(scores, chunks), reverse=True, key=lambda item: item[0])
        return [self._to_retrieved(chunk, score, "keyword") for score, chunk in ranked[:top_k] if score > 0]

    def overview_search(self, db, user_id, query, document_ids=None):
        selection = select(Document).where(Document.user_id == user_id, Document.status == "indexed")
        if document_ids:
            selection = selection.where(Document.document_id.in_(document_ids))
        document = db.scalar(selection.order_by(Document.created_at.desc(), Document.document_id).limit(1))
        if document is None:
            return []
        scope = [document.document_id]
        chunks = sorted(self._chunks(db, user_id, scope), key=lambda chunk: chunk.chunk_index)
        if not chunks:
            return []
        starts = [self._to_retrieved(chunk, 0, "document_start") for chunk in chunks[:2]]
        semantic = self.semantic_search(db, user_id, query, 4, scope, chunks=chunks)
        unique = {item.chunk_id: item for item in starts + semantic}
        self.search_diagnostics = {"overview_document_id": document.document_id,
            "first_chunk_ids": [item.chunk_id for item in starts],
            "semantic_chunk_ids": [item.chunk_id for item in semantic]}
        return self._expand_units(db, list(unique.values()), {chunk.chunk_id: chunk for chunk in chunks})

    def hybrid_search(self, db: Session, user_id: str, query: str, top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        started = time.perf_counter()
        chunks = self._chunks(db, user_id, document_ids)
        semantic = self.semantic_search(db, user_id, query, top_k, document_ids, chunks=chunks)
        semantic_end = time.perf_counter()
        keyword = self.keyword_search(db, user_id, query, top_k, document_ids, chunks=chunks)
        results = reciprocal_rank_fusion([semantic, keyword])[:top_k]
        results = self._expand_units(db, results, {chunk.chunk_id: chunk for chunk in chunks})
        from app.generation.grounding import passage_candidates, passage_key
        candidates = passage_candidates(results)[:48]
        if candidates:
            vectors = self.embedding_model.embed([query] + [quote for _, quote, _ in candidates])
            for (item, quote, _), vector in zip(candidates, vectors[1:]):
                item.metadata.setdefault("passage_similarities", {})[passage_key(quote)] = cosine_similarity(vectors[0], vector)
        self.search_diagnostics = {
            "semantic_chunk_ids": [item.chunk_id for item in semantic],
            "keyword_chunk_ids": [item.chunk_id for item in keyword],
            "semantic_ms": round((semantic_end - started) * 1000, 2),
            "keyword_fusion_ms": round((time.perf_counter() - semantic_end) * 1000, 2),
        }
        return results

    @staticmethod
    def _expand_units(db, results, chunks_by_id):
        for item in results:
            chunk = chunks_by_id[item.chunk_id]
            page = db.get(DocumentPage, f"{item.document_id}:{item.page_number or 0}")
            if page is None:
                item.metadata["index_version"] = "legacy"
                continue
            units = json.loads(page.units_json)
            matched = [u for u in units if u["start"] < chunk.end_offset and u["end"] > chunk.start_offset]
            if not matched:
                continue
            first_index = units.index(matched[0])
            if first_index and "\n\n" not in page.text[units[first_index - 1]["end"]:matched[0]["start"]]:
                matched.insert(0, units[first_index - 1])
            start = min(u["start"] for u in matched)
            end = max(u["end"] for u in matched)
            item.text = page.text[start:end]
            item.metadata["scope_context"] = "\n".join(dict.fromkeys(
                page.text[u.get("context_start", u["start"]):u.get("context_end", u["end"])].strip() for u in matched))
            item.metadata["page_source_text"] = page.text
            item.metadata["source_units"] = [dict(u, start=u["start"] - start, end=u["end"] - start) for u in matched]
            item.metadata["section"] = next((u["section"] for u in matched if u["section"]), "")
            item.metadata["index_version"] = INDEX_VERSION
        return results

    @staticmethod
    def _to_retrieved(chunk: Chunk, score: float, source: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_filename=chunk.source_filename,
            page_number=chunk.page_number,
            text=chunk.text,
            score=float(score),
            source=source,
            metadata={"chunk_index": chunk.chunk_index},
        )

