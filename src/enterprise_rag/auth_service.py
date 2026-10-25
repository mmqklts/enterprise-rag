from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from enterprise_rag.auth import hash_password, verify_password
from enterprise_rag.models import (
    KnowledgeBase,
    KnowledgeBaseMember,
    User,
)


class AuthenticationError(ValueError):
    pass


class UserAlreadyExistsError(ValueError):
    pass


@dataclass(frozen=True)
class RegisteredAccount:
    user: User
    knowledge_base: KnowledgeBase


def register_user(
    session: Session,
    username: str,
    email: str,
    password: str,
) -> RegisteredAccount:
    normalized_username = username.strip()
    normalized_email = email.strip().lower()

    existing = session.scalar(
        select(User).where(
            or_(
                User.username == normalized_username,
                User.email == normalized_email,
            )
        )
    )
    if existing is not None:
        raise UserAlreadyExistsError("用户名或邮箱已存在")

    user = User(
        username=normalized_username,
        email=normalized_email,
        password_hash=hash_password(password),
    )
    session.add(user)
    session.flush()

    knowledge_base = KnowledgeBase(
        owner_id=user.id,
        name="默认知识库",
    )
    session.add(knowledge_base)
    session.flush()

    session.add(
        KnowledgeBaseMember(
            knowledge_base_id=knowledge_base.id,
            user_id=user.id,
            role="owner",
        )
    )
    session.commit()
    session.refresh(user)
    session.refresh(knowledge_base)

    return RegisteredAccount(
        user=user,
        knowledge_base=knowledge_base,
    )


def authenticate_user(
    session: Session,
    username: str,
    password: str,
) -> User:
    user = session.scalar(
        select(User).where(User.username == username.strip())
    )
    if user is None or not verify_password(password, user.password_hash):
        raise AuthenticationError("用户名或密码错误")

    return user


def list_user_knowledge_bases(
    session: Session,
    user_id: int,
) -> list[tuple[KnowledgeBase, str]]:
    rows = session.execute(
        select(KnowledgeBase, KnowledgeBaseMember.role)
        .join(
            KnowledgeBaseMember,
            KnowledgeBaseMember.knowledge_base_id == KnowledgeBase.id,
        )
        .where(KnowledgeBaseMember.user_id == user_id)
        .order_by(KnowledgeBase.id)
    ).all()

    return [(row[0], row[1]) for row in rows]
