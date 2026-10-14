import hashlib
import json
from dataclasses import asdict
import math
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ingestion.chunker import SentenceAwareSplitter, TextChunk
from app.ingestion.langchain_documents import StructuredDocumentLoader
from app.core.errors import RAGServiceError
from app.retrieval.service import shared_embedding_model
from app.ingestion.profile import analyze, INDEX_VERSION
from app.ingestion.structure import source_units
from app.retrieval.vector_store import ChromaVectorStore
from app.storage.models import Chunk, Document, DocumentIndex, DocumentPage, User

ALLOWED_TYPES = {".pdf", ".txt", ".text", ".md", ".markdown", ".docx", ".csv", ".json"}
logger = logging.getLogger(__name__)


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


async def ingest_upload(db: Session, user: User, upload: UploadFile, progress=None) -> Document:
    progress = progress or (lambda stage: None)
    progress("Uploading file")
    settings = get_settings()
    filename = safe_filename(upload.filename or "")
    suffix = Path(filename).suffix.lower()
    logger.info("[RAG-UPLOAD] received user_id=%s format=%s", user.user_id, suffix)
    if suffix not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await upload.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")

    file_hash = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.user_id == user.user_id, Document.file_hash == file_hash))
    if existing:
        if existing.status == "indexed" and existing.index_info and json.loads(existing.index_info.analysis_json).get("index_version") == INDEX_VERSION:
            return existing
        file_path = settings.upload_dir / user.user_id / f"{existing.document_id}{suffix}"
        file_path.write_bytes(data)
        return await run_in_threadpool(index_document, db, existing, file_path, progress)

    document_id = str(uuid4())
    user_dir = settings.upload_dir / user.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{document_id}{suffix}"
    file_path.write_bytes(data)

    document = Document(
        document_id=document_id,
        user_id=user.user_id,
        filename=filename,
        file_type=suffix,
        file_hash=file_hash,
        status="processing",
    )
    db.add(document)
    db.commit()
    return await run_in_threadpool(index_document, db, document, file_path, progress)


def index_document(db: Session, document: Document, file_path: Path, progress=None) -> Document:
    progress = progress or (lambda stage: None)
    settings = get_settings()
    old_ids = [chunk.chunk_id for chunk in document.chunks]
    old_status = document.status
    new_ids = []
    store = None
    try:
        progress("Extracting text")
        loader = StructuredDocumentLoader(file_path, document.document_id, document.filename, document.file_type)
        source_documents = loader.load()
        pages = loader.pages(source_documents)
        if not any(page.text.strip() for page in pages):
            raise ValueError("Document contains no extractable text")
        logger.info("extraction complete document_id=%s pages=%d", document.document_id, len(pages))
        if sum(len(page.text) for page in pages) > settings.max_document_chars:
            raise ValueError("Document exceeds the configured extraction size limit")
        progress("Chunking")
        embedding = shared_embedding_model(settings.embedding_model)
        model = embedding._load_model()
        counter = lambda text: len(model.tokenizer.encode(text, add_special_tokens=False, verbose=False))
        max_input = model.max_seq_length
        analysis = analyze(pages, counter, settings.chunk_target_tokens, settings.chunk_overlap_tokens, max_input)
        analysis["embedding_model"] = settings.embedding_model
        analysis["embedding_mode"] = "langchain_huggingface"
        split = SentenceAwareSplitter(analysis["target_tokens"], analysis["overlap_tokens"], counter)
        chunks = [TextChunk(d.metadata["chunk_id"], d.metadata["page_number"], d.metadata["chunk_index"],
                           d.metadata["start_offset"], d.metadata["end_offset"], d.page_content, d.metadata["token_count"], d.metadata["section"])
                  for d in split.split_documents(source_documents)]
        if not chunks:
            raise ValueError("Document contains no usable content")
        logger.info("chunk count document_id=%s count=%d", document.document_id, len(chunks))
        progress("Creating embeddings")
        vectors = embedding.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks) or not all(vector and all(math.isfinite(v) for v in vector) for vector in vectors):
            raise ValueError("Embedding output is incomplete or invalid")
        analysis["chunk_count"] = len(chunks)
        logger.info("embedding dimension document_id=%s dimension=%d", document.document_id, len(vectors[0]))
        analysis["max_chunk_tokens"] = max(chunk.token_count for chunk in chunks)
        new_ids = [chunk.chunk_id for chunk in chunks]
        # Keep existing SQL data intact until extraction and embeddings have succeeded.
        db.query(Chunk).filter(Chunk.document_id == document.document_id).delete(synchronize_session="fetch")
        db.query(DocumentPage).filter(DocumentPage.document_id == document.document_id).delete(synchronize_session="fetch")
        db.flush()
        for page in pages:
            db.add(DocumentPage(page_id=f"{document.document_id}:{page.page_number or 0}",
                                document_id=document.document_id, page_number=page.page_number,
                                text=page.text, units_json=json.dumps([asdict(u) for u in source_units(page.text, pdf=page.page_number is not None)])))
        db_chunks: list[Chunk] = []
        for chunk, vector in zip(chunks, vectors):
            db_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                document_id=document.document_id,
                user_id=document.user_id,
                source_filename=document.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                text=chunk.text,
                embedding_json=json.dumps(vector),
            )
            db_chunk.source_section = chunk.section
            db.add(db_chunk)
            db_chunks.append(db_chunk)
        db.flush()
        progress("Indexing in ChromaDB")
        store = ChromaVectorStore(embedding)
        store.upsert_chunks(db_chunks, vectors)
        logger.info("[RAG-INDEX] Chroma complete document_id=%s count=%d", document.document_id, len(chunks))
        analysis["vector_index"] = "langchain_chroma"
        info = db.get(DocumentIndex, document.document_id)
        if info is None:
            info = DocumentIndex(document_id=document.document_id, analysis_json="{}")
            db.add(info)
        info.analysis_json = json.dumps(analysis)
        document.status = "indexed"
        document.error_message = None
        db.commit()
        try:
            store.delete_chunks(list(set(old_ids) - set(new_ids)))
        except Exception as exc:
            analysis["warnings"].append({"page": None, "message": f"Old vector cleanup failed: {type(exc).__name__}"})
            info.analysis_json = json.dumps(analysis)
            db.commit()
        db.expire(document, ["chunks", "index_info", "pages"])
        db.refresh(document)
        return document
    except Exception as exc:
        logger.warning("ingestion failed document_id=%s error_type=%s", document.document_id, type(exc).__name__)
        db.rollback()
        if store is not None and set(new_ids) - set(old_ids):
            try:
                store.delete_chunks(list(set(new_ids) - set(old_ids)))
            except Exception:
                logging.getLogger(__name__).warning("Could not clean up vectors after failed ingestion")
        document.status = "indexed" if old_status == "indexed" else "failed"
        document.error_message = str(exc)
        db.commit()
        if isinstance(exc, RAGServiceError):
            raise
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

