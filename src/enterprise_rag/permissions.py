from collections.abc import Collection

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_rag.models import KnowledgeBase, KnowledgeBaseMember


def require_knowledge_base_access(
    session: Session,
    user_id: int,
    knowledge_base_id: int,
    allowed_roles: Collection[str],
) -> KnowledgeBase:
    membership = session.scalar(
        select(KnowledgeBaseMember).where(
            KnowledgeBaseMember.user_id == user_id,
            KnowledgeBaseMember.knowledge_base_id == knowledge_base_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=403, detail="无权访问该知识库")

    if membership.role not in allowed_roles:
        raise HTTPException(status_code=403, detail="当前角色无此操作权限")

    knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    return knowledge_base
