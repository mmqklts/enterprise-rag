import importlib
from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db

main = importlib.import_module("enterprise_rag.main")


def test_create_and_list_knowledge_bases() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)

    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app = main.app
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            register = client.post(
                "/auth/register",
                json={
                    "username": "kb-user",
                    "email": "kb-user@example.com",
                    "password": "strong-password",
                },
            )
            token = register.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            created = client.post(
                "/knowledge-bases",
                headers=headers,
                json={"name": "研发知识库"},
            )
            listed = client.get(
                "/knowledge-bases",
                headers=headers,
            )

            assert created.status_code == 201
            assert created.json()["name"] == "研发知识库"
            assert created.json()["role"] == "owner"

            assert listed.status_code == 200
            assert [item["name"] for item in listed.json()] == [
                "默认知识库",
                "研发知识库",
            ]
    finally:
        app.dependency_overrides.clear()