from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
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
from enterprise_rag.observability import audit_log
from enterprise_rag.permissions import require_knowledge_base_access
from enterprise_rag.rate_limit import enforce_rate_limit, env_int

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
    http_request: Request,
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

    audit_log(
        event="knowledge_base",
        action="member_add",
        outcome="success",
        request_id=http_request.state.request_id,
        actor_user_id=user.id,
        resource_type="knowledge_base",
        resource_id=knowledge_base_id,
        target_user_id=target_user.id,
        role=membership.role,
    )
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
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> KnowledgeBaseResponse:
    enforce_rate_limit(
        http_request,
        bucket="knowledge-base-create",
        limit=env_int("RATE_LIMIT_KB_CREATE_PER_HOUR", 10),
        window_seconds=3600,
        subject=f"user:{user.id}",
    )
    try:
        knowledge_base = create_knowledge_base(
            session=session,
            owner_id=user.id,
            name=request.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    audit_log(
        event="knowledge_base",
        action="create",
        outcome="success",
        request_id=http_request.state.request_id,
        actor_user_id=user.id,
        resource_type="knowledge_base",
        resource_id=knowledge_base.id,
    )
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
    http_request: Request,
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

    audit_log(
        event="knowledge_base",
        action="member_role_update",
        outcome="success",
        request_id=http_request.state.request_id,
        actor_user_id=user.id,
        resource_type="knowledge_base",
        resource_id=knowledge_base_id,
        target_user_id=target_user.id,
        role=membership.role,
    )
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
    http_request: Request,
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

    audit_log(
        event="knowledge_base",
        action="member_remove",
        outcome="success",
        request_id=http_request.state.request_id,
        actor_user_id=user.id,
        resource_type="knowledge_base",
        resource_id=knowledge_base_id,
        target_user_id=user_id,
    )
