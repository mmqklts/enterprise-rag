from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from enterprise_rag.models import (
    KnowledgeBase,
    KnowledgeBaseMember,
    User,
)

VALID_KNOWLEDGE_BASE_ROLES = {"owner", "editor", "viewer"}

def create_knowledge_base(
    session: Session,
    owner_id: int,
    name: str,
) -> KnowledgeBase:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("知识库名称不能为空")

    knowledge_base = KnowledgeBase(
        owner_id=owner_id,
        name=normalized_name,
    )
    session.add(knowledge_base)

    try:
        session.flush()
        session.add(
            KnowledgeBaseMember(
                knowledge_base_id=knowledge_base.id,
                user_id=owner_id,
                role="owner",
            )
        )
        session.commit()
        session.refresh(knowledge_base)
        return knowledge_base
    except IntegrityError as exc:
        session.rollback()
        raise ValueError("同名知识库已经存在") from exc
    except Exception:
        session.rollback()
        raise

def add_knowledge_base_member(
    session: Session,
    knowledge_base_id: int,
    user_id: int,
    role: str,
) -> KnowledgeBaseMember:
    if role not in VALID_KNOWLEDGE_BASE_ROLES:
        raise ValueError("无效的知识库角色")

    knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
    user = session.get(User, user_id)
    if knowledge_base is None:
        raise LookupError("知识库不存在")
    if user is None:
        raise LookupError("用户不存在")

    existing = session.scalar(
        select(KnowledgeBaseMember).where(
            KnowledgeBaseMember.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseMember.user_id == user_id,
        )
    )
    if existing is not None:
        raise ValueError("该用户已经是知识库成员")

    membership = KnowledgeBaseMember(
        knowledge_base_id=knowledge_base_id,
        user_id=user_id,
        role=role,
    )
    session.add(membership)

    try:
        session.commit()
        session.refresh(membership)
        return membership
    except IntegrityError as exc:
        session.rollback()
        raise ValueError("该用户已经是知识库成员") from exc

def update_knowledge_base_member_role(
    session: Session,
    knowledge_base_id: int,
    user_id: int,
    role: str,
) -> KnowledgeBaseMember:
    if role not in VALID_KNOWLEDGE_BASE_ROLES:
        raise ValueError("无效的知识库角色")

    membership = session.scalar(
        select(KnowledgeBaseMember).where(
            KnowledgeBaseMember.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseMember.user_id == user_id,
        )
    )
    if membership is None:
        raise LookupError("知识库成员不存在")

    if membership.role == "owner" and role != "owner":
        owner_count = session.scalar(
            select(func.count())
            .select_from(KnowledgeBaseMember)
            .where(
                KnowledgeBaseMember.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMember.role == "owner",
            )
        )
        if owner_count is not None and owner_count <= 1:
            raise ValueError("不能降级最后一个 owner")

    membership.role = role

    try:
        session.commit()
        session.refresh(membership)
        return membership
    except Exception:
        session.rollback()
        raise


def remove_knowledge_base_member(
    session: Session,
    knowledge_base_id: int,
    user_id: int,
) -> None:
    membership = session.scalar(
        select(KnowledgeBaseMember).where(
            KnowledgeBaseMember.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseMember.user_id == user_id,
        )
    )
    if membership is None:
        raise LookupError("知识库成员不存在")

    if membership.role == "owner":
        owner_count = session.scalar(
            select(func.count())
            .select_from(KnowledgeBaseMember)
            .where(
                KnowledgeBaseMember.knowledge_base_id
                == knowledge_base_id,
                KnowledgeBaseMember.role == "owner",
            )
        )
        if owner_count is not None and owner_count <= 1:
            raise ValueError("不能移除最后一个 owner")

    session.delete(membership)

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise