from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_rag.auth_routes import get_current_user
from enterprise_rag.auth_service import list_user_knowledge_bases
from enterprise_rag.db import get_db
from enterprise_rag.knowledge_base_service import (
    add_knowledge_base_member,
    create_knowledge_base,
    remove_knowledge_base_member,
    update_knowledge_base_member_role,
)
from enterprise_rag.models import User
from enterprise_rag.permissions import require_knowledge_base_access

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["knowledge-bases"],
)


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class KnowledgeBaseResponse(BaseModel):
    id: int
    name: str
    role: str

class KnowledgeBaseMemberCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: str = Field(pattern=r"^(owner|editor|viewer)$")


class KnowledgeBaseMemberResponse(BaseModel):
    id: int
    knowledge_base_id: int
    user_id: int
    username: str
    role: str

class KnowledgeBaseMemberRoleUpdateRequest(BaseModel):
    role: str = Field(pattern=r"^(owner|editor|viewer)$")

@router.post(
    "/{knowledge_base_id}/members",
    response_model=KnowledgeBaseMemberResponse,
    status_code=201,
)
def add_member(
    knowledge_base_id: int,
    request: KnowledgeBaseMemberCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> KnowledgeBaseMemberResponse:
    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner"},
    )
    target_user = session.scalar(
        select(User).where(User.username == request.username.strip())
    )
    if target_user is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    try:
        membership = add_knowledge_base_member(
            session=session,
            knowledge_base_id=knowledge_base_id,
            user_id=target_user.id,
            role=request.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return KnowledgeBaseMemberResponse(
        id=membership.id,
        knowledge_base_id=membership.knowledge_base_id,
        user_id=membership.user_id,
        username=target_user.username,
        role=membership.role,
    )

@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
def create(
    request: KnowledgeBaseCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> KnowledgeBaseResponse:
    try:
        knowledge_base = create_knowledge_base(
            session=session,
            owner_id=user.id,
            name=request.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return KnowledgeBaseResponse(
        id=knowledge_base.id,
        name=knowledge_base.name,
        role="owner",
    )


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_all(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> list[KnowledgeBaseResponse]:
    knowledge_bases = list_user_knowledge_bases(session, user.id)
    return [
        KnowledgeBaseResponse(
            id=knowledge_base.id,
            name=knowledge_base.name,
            role=role,
        )
        for knowledge_base, role in knowledge_bases
    ]

@router.patch(
    "/{knowledge_base_id}/members/{user_id}",
    response_model=KnowledgeBaseMemberResponse,
)
def update_member_role(
    knowledge_base_id: int,
    user_id: int,
    request: KnowledgeBaseMemberRoleUpdateRequest,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> KnowledgeBaseMemberResponse:
    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner"},
    )

    try:
        membership = update_knowledge_base_member_role(
            session=session,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            role=request.role,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    target_user = session.get(User, user_id)
    if target_user is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    return KnowledgeBaseMemberResponse(
        id=membership.id,
        knowledge_base_id=membership.knowledge_base_id,
        user_id=membership.user_id,
        username=target_user.username,
        role=membership.role,
    )


@router.delete(
    "/{knowledge_base_id}/members/{user_id}",
    status_code=204,
)
def delete_member(
    knowledge_base_id: int,
    user_id: int,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> None:
    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner"},
    )

    try:
        remove_knowledge_base_member(
            session=session,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc