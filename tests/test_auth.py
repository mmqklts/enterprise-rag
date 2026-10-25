import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag.auth import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from enterprise_rag.auth_service import (
    AuthenticationError,
    UserAlreadyExistsError,
    authenticate_user,
    list_user_knowledge_bases,
    register_user,
)
from enterprise_rag.db import Base


def create_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_password_hashing_and_jwt() -> None:
    encoded = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)

    token = create_access_token(user_id=7, username="alice")
    payload = decode_access_token(token)
    assert payload["sub"] == "7"
    assert payload["username"] == "alice"

    header, body, signature = token.split(".")
    with pytest.raises(AuthError):
        decode_access_token(f"{header}.{body}.x{signature[1:]}")


def test_register_login_and_default_knowledge_base() -> None:
    with create_session() as session:
        account = register_user(
            session=session,
            username="alice",
            email="alice@example.com",
            password="strong-password",
        )

        assert account.user.username == "alice"
        assert account.knowledge_base.name == "默认知识库"

        user = authenticate_user(
            session=session,
            username="alice",
            password="strong-password",
        )
        knowledge_bases = list_user_knowledge_bases(session, user.id)

        assert user.id == account.user.id
        assert knowledge_bases[0][0].id == account.knowledge_base.id
        assert knowledge_bases[0][1] == "owner"

        with pytest.raises(AuthenticationError):
            authenticate_user(
                session=session,
                username="alice",
                password="wrong-password",
            )

        with pytest.raises(UserAlreadyExistsError):
            register_user(
                session=session,
                username="alice",
                email="other@example.com",
                password="another-password",
            )
