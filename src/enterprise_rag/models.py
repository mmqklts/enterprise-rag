from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from enterprise_rag.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    memberships: Mapped[list["KnowledgeBaseMember"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "name",
            name="uq_knowledge_bases_owner_name",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    owner: Mapped[User] = relationship(back_populates="knowledge_bases")
    members: Mapped[list["KnowledgeBaseMember"]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )


class KnowledgeBaseMember(Base):
    __tablename__ = "knowledge_base_members"
    __table_args__ = (
        CheckConstraint(
            "role IN ('owner', 'editor', 'viewer')",
            name="ck_knowledge_base_members_role",
        ),
        UniqueConstraint(
            "knowledge_base_id",
            "user_id",
            name="uq_knowledge_base_members_user",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="members",
    )
    user: Mapped[User] = relationship(back_populates="memberships")
