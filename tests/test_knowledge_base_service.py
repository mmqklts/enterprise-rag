import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag.db import Base
from enterprise_rag.knowledge_base_service import create_knowledge_base
from enterprise_rag.models import KnowledgeBaseMember, User


def test_create_knowledge_base_with_owner_membership() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        owner = User(
            username="owner",
            email="owner@example.com",
            password_hash="hash",
        )
        session.add(owner)
        session.commit()

        knowledge_base = create_knowledge_base(
            session=session,
            owner_id=owner.id,
            name="内部制度",
        )

        membership = session.query(KnowledgeBaseMember).one()
        assert knowledge_base.owner_id == owner.id
        assert knowledge_base.name == "内部制度"
        assert membership.knowledge_base_id == knowledge_base.id
        assert membership.user_id == owner.id
        assert membership.role == "owner"

        with pytest.raises(ValueError):
            create_knowledge_base(
                session=session,
                owner_id=owner.id,
                name="内部制度",
            )

def test_add_knowledge_base_member() -> None:
    from enterprise_rag.knowledge_base_service import (
        add_knowledge_base_member,
    )
    from enterprise_rag.models import User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        owner = User(
            username="member-owner",
            email="member-owner@example.com",
            password_hash="hash",
        )
        target = User(
            username="member-viewer",
            email="member-viewer@example.com",
            password_hash="hash",
        )
        session.add_all([owner, target])
        session.commit()

        knowledge_base = create_knowledge_base(
            session=session,
            owner_id=owner.id,
            name="成员测试知识库",
        )
        membership = add_knowledge_base_member(
            session=session,
            knowledge_base_id=knowledge_base.id,
            user_id=target.id,
            role="viewer",
        )

        assert membership.user_id == target.id
        assert membership.knowledge_base_id == knowledge_base.id
        assert membership.role == "viewer"

        with pytest.raises(ValueError):
            add_knowledge_base_member(
                session=session,
                knowledge_base_id=knowledge_base.id,
                user_id=target.id,
                role="viewer",
            )