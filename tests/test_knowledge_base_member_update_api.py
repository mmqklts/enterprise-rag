import importlib
from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db

main = importlib.import_module("enterprise_rag.main")


def test_update_and_delete_member_and_protect_last_owner() -> None:
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
            def register(username: str) -> tuple[str, int]:
                response = client.post(
                    "/auth/register",
                    json={
                        "username": username,
                        "email": f"{username}@example.com",
                        "password": "strong-password",
                    },
                )
                assert response.status_code == 200
                data = response.json()
                return data["access_token"], data["user_id"]

            owner_token, owner_id = register("update-owner")
            member_token, member_id = register("update-member")
            owner_headers = {
                "Authorization": f"Bearer {owner_token}",
            }
            member_headers = {
                "Authorization": f"Bearer {member_token}",
            }

            created = client.post(
                "/knowledge-bases",
                headers=owner_headers,
                json={"name": "角色测试知识库"},
            )
            knowledge_base_id = created.json()["id"]

            added = client.post(
                f"/knowledge-bases/{knowledge_base_id}/members",
                headers=owner_headers,
                json={"username": "update-member", "role": "viewer"},
            )
            assert added.status_code == 201

            updated = client.patch(
                f"/knowledge-bases/{knowledge_base_id}/members/{member_id}",
                headers=owner_headers,
                json={"role": "editor"},
            )
            assert updated.status_code == 200
            assert updated.json()["role"] == "editor"

            listed = client.get(
                "/knowledge-bases",
                headers=member_headers,
            )
            role_by_id = {
                item["id"]: item["role"]
                for item in listed.json()
            }
            assert role_by_id[knowledge_base_id] == "editor"

            deleted = client.delete(
                f"/knowledge-bases/{knowledge_base_id}/members/{member_id}",
                headers=owner_headers,
            )
            assert deleted.status_code == 204

            listed_after_delete = client.get(
                "/knowledge-bases",
                headers=member_headers,
            )
            assert all(
                item["id"] != knowledge_base_id
                for item in listed_after_delete.json()
            )

            demote_owner = client.patch(
                f"/knowledge-bases/{knowledge_base_id}/members/{owner_id}",
                headers=owner_headers,
                json={"role": "viewer"},
            )
            assert demote_owner.status_code == 409

            delete_owner = client.delete(
                f"/knowledge-bases/{knowledge_base_id}/members/{owner_id}",
                headers=owner_headers,
            )
            assert delete_owner.status_code == 409
    finally:
        app.dependency_overrides.clear()