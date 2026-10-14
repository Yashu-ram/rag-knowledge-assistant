from pydantic import BaseModel, EmailStr, Field, model_validator
from typing import Literal


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    full_name: str
    email: str


class UserMe(BaseModel):
    user_id: str
    full_name: str
    email: str


class DocumentResponse(BaseModel):
    error_message: str | None = None
    document_id: str
    filename: str
    file_type: str
    status: str
    created_at: str
    analysis: dict = Field(default_factory=dict)


class Citation(BaseModel):
    section: str | None = None
    citation_id: int | None = None
    anchor: str | None = None
    context_excerpt: str | None = None
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int | None = None
    excerpt: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    model_name: Literal["qwen2.5:1.5b", "qwen2.5:3b"] | None = None
    response_mode: Literal["quick", "deep"] = "quick"
    document_ids: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_single_document_and_model_alias(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for alias, field in (("document_id", "document_ids"), ("model", "model_name")):
            if alias not in value:
                continue
            converted = [value[alias]] if alias == "document_id" and value[alias] is not None else value[alias]
            if field in value and value[field] != converted:
                raise ValueError(f"Conflicting {alias} and {field}")
            value[field] = converted
            value.pop(alias)
        return value


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    message_id: str
    model_used: str | None
    answerable: str
    citations: list[Citation]
    retrieval_metadata: dict
    latency_ms: int


class ConversationCreate(BaseModel):
    title: str = "New Chat"


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)

class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str

