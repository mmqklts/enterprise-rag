from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from enterprise_rag.auth import (
    AuthError,
    create_access_token,
    decode_access_token,
)
from enterprise_rag.auth_service import (
    AuthenticationError,
    UserAlreadyExistsError,
    authenticate_user,
    list_user_knowledge_bases,
    register_user,
)
from enterprise_rag.db import get_db
from enterprise_rag.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    email: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class KnowledgeBaseSummary(BaseModel):
    id: int
    name: str
    role: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    knowledge_bases: list[KnowledgeBaseSummary]


class CurrentUserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    knowledge_bases: list[KnowledgeBaseSummary]


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    session: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少认证 Token")

    try:
        payload = decode_access_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = session.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def _auth_response(
    session: Session,
    user: User,
) -> AuthResponse:
    knowledge_bases = list_user_knowledge_bases(session, user.id)
    return AuthResponse(
        access_token=create_access_token(user.id, user.username),
        user_id=user.id,
        username=user.username,
        knowledge_bases=[
            KnowledgeBaseSummary(
                id=knowledge_base.id,
                name=knowledge_base.name,
                role=role,
            )
            for knowledge_base, role in knowledge_bases
        ],
    )


@router.post("/register", response_model=AuthResponse)
def register(
    request: RegisterRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AuthResponse:
    try:
        account = register_user(
            session=session,
            username=request.username,
            email=request.email,
            password=request.password,
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _auth_response(session, account.user)


@router.post("/login", response_model=AuthResponse)
def login(
    request: LoginRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AuthResponse:
    try:
        user = authenticate_user(
            session=session,
            username=request.username,
            password=request.password,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    return _auth_response(session, user)


@router.get("/me", response_model=CurrentUserResponse)
def me(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> CurrentUserResponse:
    knowledge_bases = list_user_knowledge_bases(session, user.id)
    return CurrentUserResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        knowledge_bases=[
            KnowledgeBaseSummary(
                id=knowledge_base.id,
                name=knowledge_base.name,
                role=role,
            )
            for knowledge_base, role in knowledge_bases
        ],
    )
