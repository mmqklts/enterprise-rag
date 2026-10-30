import importlib
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db
from enterprise_rag.document_service import upsert_document_record

main = importlib.import_module("enterprise_rag.main")


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def delete_by_document_id(
        self,
        document_id: str,
        knowledge_base_id: str = "default",
    ) -> None:
        self.deleted.append((document_id, knowledge_base_id))


def test_document_list_and_delete_sync_file_and_vectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    vector_store = FakeVectorStore()
    monkeypatch.setattr(main, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    app = main.app
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            def register(username: str) -> dict:
                response = client.post(
                    "/auth/register",
                    json={
                        "username": username,
                        "email": f"{username}@example.com",
                        "password": "strong-password",
                    },
                )
                assert response.status_code == 200
                return response.json()

            owner = register("document-owner")
            outsider = register("document-outsider")
            knowledge_base_id = owner["knowledge_bases"][0]["id"]
            owner_headers = {
                "Authorization": f"Bearer {owner['access_token']}",
            }
            outsider_headers = {
                "Authorization": f"Bearer {outsider['access_token']}",
            }

            stored_file = tmp_path / "doc-1.txt"
            stored_file.write_text("文档测试", encoding="utf-8")
            with session_factory() as session:
                upsert_document_record(
                    session=session,
                    knowledge_base_id=knowledge_base_id,
                    created_by=owner["user_id"],
                    document_id="doc-1",
                    filename="doc-1.txt",
                    stored_path=str(stored_file),
                    chunking_strategy="heading",
                    chunk_count=2,
                )

            listed = client.get(
                f"/knowledge-bases/{knowledge_base_id}/documents",
                headers=owner_headers,
            )
            assert listed.status_code == 200
            assert listed.json()[0]["document_id"] == "doc-1"

            forbidden = client.get(
                f"/knowledge-bases/{knowledge_base_id}/documents",
                headers=outsider_headers,
            )
            assert forbidden.status_code == 403

            deleted = client.delete(
                f"/knowledge-bases/{knowledge_base_id}/documents/doc-1",
                headers=owner_headers,
            )
            assert deleted.status_code == 204
            assert not stored_file.exists()
            assert vector_store.deleted == [
                ("doc-1", str(knowledge_base_id))
            ]

            listed_after = client.get(
                f"/knowledge-bases/{knowledge_base_id}/documents",
                headers=owner_headers,
            )
            assert listed_after.json() == []
    finally:
        app.dependency_overrides.clear()
