from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag.db import Base
from enterprise_rag.models import (
    KnowledgeBase,
    KnowledgeBaseMember,
    User,
)


def test_user_knowledge_base_and_member_relationships() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(
            username="alice",
            email="alice@example.com",
            password_hash="hashed-password",
        )
        knowledge_base = KnowledgeBase(
            owner=user,
            name="内部制度",
        )
        membership = KnowledgeBaseMember(
            knowledge_base=knowledge_base,
            user=user,
            role="owner",
        )
        session.add_all([user, knowledge_base, membership])
        session.commit()

        assert user.knowledge_bases[0].name == "内部制度"
        assert knowledge_base.owner.username == "alice"
        assert knowledge_base.members[0].role == "owner"
        assert membership.user.email == "alice@example.com"
