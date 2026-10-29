import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag.auth_service import register_user
from enterprise_rag.db import Base
from enterprise_rag.models import KnowledgeBaseMember
from enterprise_rag.permissions import require_knowledge_base_access


def create_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_knowledge_base_roles_are_enforced() -> None:
    with create_session() as session:
        owner_account = register_user(
            session=session,
            username="owner",
            email="owner@example.com",
            password="strong-password",
        )
        viewer_account = register_user(
            session=session,
            username="viewer",
            email="viewer@example.com",
            password="strong-password",
        )
        outsider_account = register_user(
            session=session,
            username="outsider",
            email="outsider@example.com",
            password="strong-password",
        )
        session.add(
            KnowledgeBaseMember(
                knowledge_base_id=owner_account.knowledge_base.id,
                user_id=viewer_account.user.id,
                role="viewer",
            )
        )
        session.commit()

        knowledge_base = require_knowledge_base_access(
            session=session,
            user_id=viewer_account.user.id,
            knowledge_base_id=owner_account.knowledge_base.id,
            allowed_roles={"owner", "editor", "viewer"},
        )
        assert knowledge_base.id == owner_account.knowledge_base.id

        with pytest.raises(HTTPException) as upload_error:
            require_knowledge_base_access(
                session=session,
                user_id=viewer_account.user.id,
                knowledge_base_id=owner_account.knowledge_base.id,
                allowed_roles={"owner", "editor"},
            )
        assert upload_error.value.status_code == 403

        with pytest.raises(HTTPException) as outsider_error:
            require_knowledge_base_access(
                session=session,
                user_id=outsider_account.user.id,
                knowledge_base_id=owner_account.knowledge_base.id,
                allowed_roles={"owner", "editor", "viewer"},
            )
        assert outsider_error.value.status_code == 403
