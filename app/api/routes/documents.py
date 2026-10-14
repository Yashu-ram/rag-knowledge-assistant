from fastapi import APIRouter, Depends, HTTPException, UploadFile
import json
import asyncio
import logging
import io
from fastapi.responses import StreamingResponse
from app.core.errors import RAGServiceError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import DocumentResponse
from app.auth.security import get_current_user
from app.ingestion.pipeline import ingest_upload, index_document
from app.core.config import get_settings
from app.retrieval.vector_store import ChromaVectorStore
from app.storage.database import get_db
from app.storage.models import Document, User
from app.storage.document_access import SAMPLE_LIBRARY_USER_ID, readable_owner_ids, can_read_document

router = APIRouter(prefix="/documents", tags=["documents"])


def to_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        is_sample=document.user_id == SAMPLE_LIBRARY_USER_ID,
        error_message=document.error_message,
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        status=document.status,
        created_at=document.created_at.isoformat(),
        analysis=json.loads(document.index_info.analysis_json) if document.index_info else {"index_version": "legacy"},
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    stream: bool = False,
):
    if stream:
        # Streaming/disconnects must not close the worker's DB or upload handles.
        data = await file.read(get_settings().max_upload_bytes + 1)
        filename, user_id, bind = file.filename, user.user_id, db.get_bind()
        async def events():
            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            def progress(stage):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "stage": stage})
            async def work():
                try:
                    with Session(bind=bind) as worker_db:
                        upload = UploadFile(filename=filename, file=io.BytesIO(data))
                        try:
                            document = await ingest_upload(worker_db, worker_db.get(User, user_id), upload, progress)
                            await queue.put({"type": "result", "document": to_document_response(document).model_dump()})
                        finally:
                            await upload.close()
                except (HTTPException, RAGServiceError) as exc:
                    await queue.put({"type": "error", "message": str(exc.detail) if isinstance(exc, HTTPException) else str(exc),
                                     "status_code": exc.status_code, "code": getattr(exc, "code", "INGESTION_FAILED")})
                except Exception:
                    logging.getLogger(__name__).exception("Document upload failed")
                    await queue.put({"type": "error", "message": "Document ingestion failed. Check the backend logs.", "status_code": 500})
            task = asyncio.create_task(work())
            try:
                while True:
                    event = await queue.get()
                    yield json.dumps(event) + "\n"
                    if event["type"] in {"result", "error"}:
                        break
            finally:
                # Finish the bounded ingestion before request-scoped DB/file handles close.
                await asyncio.shield(task)
        return StreamingResponse(events(), media_type="application/x-ndjson")
    document = await ingest_upload(db, user, file)
    return to_document_response(document)


@router.get("", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[DocumentResponse]:
    documents = db.scalars(select(Document).where(Document.user_id.in_(readable_owner_ids(user.user_id))).order_by(Document.created_at.desc())).all()
    return [to_document_response(document) for document in documents]


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DocumentResponse:
    document = db.get(Document, document_id)
    if document is None or document.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    path = get_settings().upload_dir / user.user_id / f"{document.document_id}{document.file_type}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original uploaded file is unavailable")
    return to_document_response(index_document(db, document, path))


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DocumentResponse:
    document = db.get(Document, document_id)
    if not can_read_document(document, user.user_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return to_document_response(document)


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    document = db.get(Document, document_id)
    if document is None or document.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    chunk_ids = [chunk.chunk_id for chunk in document.chunks]
    ChromaVectorStore().delete_chunks(chunk_ids)
    db.delete(document)
    db.commit()
    return {"ok": True}

