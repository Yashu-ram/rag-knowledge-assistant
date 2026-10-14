import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import ConversationCreate, ConversationRename, ConversationResponse
from app.auth.security import get_current_user
from app.storage.database import get_db
from app.storage.models import Conversation, Message, User

router = APIRouter(prefix="/conversations", tags=["conversations"])


def to_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@router.post("", response_model=ConversationResponse)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ConversationResponse:
    conversation = Conversation(conversation_id=str(uuid4()), user_id=user.user_id, title=payload.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return to_response(conversation)


@router.get("", response_model=list[ConversationResponse])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ConversationResponse]:
    conversations = db.scalars(select(Conversation).where(Conversation.user_id == user.user_id).order_by(Conversation.updated_at.desc())).all()
    return [to_response(item) for item in conversations]


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())).all()
    return {
        "conversation": to_response(conversation),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "model_used": m.model_used,
                "answerable": m.answerable,
                "citations": json.loads(m.citations_json or "[]"),
            }
            for m in messages
        ],
    }


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def rename_conversation(conversation_id: str, payload: ConversationRename, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ConversationResponse:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation.title = payload.title
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(conversation)
    return to_response(conversation)


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
    return {"ok": True}

