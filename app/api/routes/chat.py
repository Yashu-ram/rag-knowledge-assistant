import json
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.schemas import ChatRequest, ChatResponse
from app.auth.security import get_current_user
from app.rag_pipeline import RAGPipeline
from app.generation.model_gateway import generate_answer, ensure_ollama_model
from app.storage.database import get_db
from app.storage.models import Conversation, Document, Message, User

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
@router.post("/query", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatResponse:
    started = time.perf_counter()
    if payload.document_ids:
        for document_id in payload.document_ids:
            document = db.get(Document, document_id)
            if document is None or document.user_id != user.user_id:
                raise HTTPException(status_code=404, detail="Document not found")
            if document.status != "indexed":
                raise HTTPException(status_code=409, detail=f"Document '{document.filename}' is {document.status}. Wait for ingestion to complete successfully before querying it.")
    elif db.scalar(select(Document.document_id).where(Document.user_id == user.user_id, Document.status == "processing")):
        raise HTTPException(status_code=409, detail="Document ingestion is still processing. Wait for it to complete before asking a question.")
    selected_model = ensure_ollama_model(payload.model_name)

    conversation = None
    if payload.conversation_id:
        conversation = db.get(Conversation, payload.conversation_id)
        if conversation is None or conversation.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = Conversation(conversation_id=str(uuid4()), user_id=user.user_id, title=payload.question[:70])
        db.add(conversation)
        db.flush()

    result = RAGPipeline(generate_answer).run(db, user.user_id, payload.question,
        payload.document_ids, selected_model, payload.response_mode)
    answer, citations = result.answer, result.citations
    model_used, answerability = result.model_used, result.answerability

    user_message = Message(
        message_id=str(uuid4()),
        conversation_id=conversation.conversation_id,
        user_id=user.user_id,
        role="user",
        content=payload.question,
    )
    assistant_message = Message(
        message_id=str(uuid4()),
        conversation_id=conversation.conversation_id,
        user_id=user.user_id,
        role="assistant",
        content=answer,
        model_used=model_used,
        answerable=answerability,
        citations_json=json.dumps([citation.model_dump() for citation in citations]),
    )
    conversation.updated_at = datetime.now(timezone.utc)
    db.add_all([user_message, assistant_message])
    db.commit()

    total_latency = int((time.perf_counter() - started) * 1000)
    return ChatResponse(
        answer=answer,
        conversation_id=conversation.conversation_id,
        message_id=assistant_message.message_id,
        model_used=model_used,
        answerable=answerability,
        citations=citations,
        retrieval_metadata=result.metadata,
        latency_ms=total_latency,
    )

