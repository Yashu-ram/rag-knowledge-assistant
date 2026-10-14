from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import AuthResponse, LoginRequest, SignupRequest, UserMe
from app.auth.security import (
    create_access_token,
    get_current_token_id,
    get_current_user,
    hash_password,
    normalize_email,
    revoke_token,
    verify_password,
)
from app.storage.database import get_db
from app.storage.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = normalize_email(payload.email)
    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = User(
        user_id=str(uuid4()),
        full_name=payload.full_name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(db, user)
    return AuthResponse(access_token=token, user_id=user.user_id, full_name=user.full_name, email=user.email)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = normalize_email(payload.email)
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(db, user)
    return AuthResponse(access_token=token, user_id=user.user_id, full_name=user.full_name, email=user.email)


@router.get("/me", response_model=UserMe)
def me(user: User = Depends(get_current_user)) -> UserMe:
    return UserMe(user_id=user.user_id, full_name=user.full_name, email=user.email)


@router.post("/logout")
def logout(
    token_id: str = Depends(get_current_token_id),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    revoke_token(db, token_id)
    return {"ok": True}

