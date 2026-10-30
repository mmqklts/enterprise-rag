import importlib
from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db

main = importlib.import_module("enterprise_rag.main")


def test_only_owner_can_add_members() -> None:
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
            def register(username: str) -> str:
                response = client.post(
                    "/auth/register",
                    json={
                        "username": username,
                        "email": f"{username}@example.com",
                        "password": "strong-password",
                    },
                )
                assert response.status_code == 200
                return response.json()["access_token"]

            owner_token = register("owner-user")
            viewer_token = register("viewer-user")
            third_token = register("third-user")

            owner_headers = {
                "Authorization": f"Bearer {owner_token}",
            }
            viewer_headers = {
                "Authorization": f"Bearer {viewer_token}",
            }
            third_headers = {
                "Authorization": f"Bearer {third_token}",
            }

            created = client.post(
                "/knowledge-bases",
                headers=owner_headers,
                json={"name": "团队知识库"},
            )
            knowledge_base_id = created.json()["id"]

            added = client.post(
                f"/knowledge-bases/{knowledge_base_id}/members",
                headers=owner_headers,
                json={"username": "viewer-user", "role": "viewer"},
            )
            assert added.status_code == 201
            assert added.json()["role"] == "viewer"
            assert added.json()["username"] == "viewer-user"

            duplicate = client.post(
                f"/knowledge-bases/{knowledge_base_id}/members",
                headers=owner_headers,
                json={"username": "viewer-user", "role": "editor"},
            )
            assert duplicate.status_code == 409

            forbidden = client.post(
                f"/knowledge-bases/{knowledge_base_id}/members",
                headers=viewer_headers,
                json={"username": "third-user", "role": "viewer"},
            )
            assert forbidden.status_code == 403

            owner_can_query = client.get(
                "/knowledge-bases",
                headers=third_headers,
            )
            assert owner_can_query.status_code == 200
            assert [item["name"] for item in owner_can_query.json()] == [
                "默认知识库"
            ]
    finally:
        app.dependency_overrides.clear()