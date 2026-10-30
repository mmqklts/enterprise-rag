import importlib
from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db
from enterprise_rag.job_service import (
    mark_job_completed,
    mark_job_processing,
)

main = importlib.import_module("enterprise_rag.main")


def test_upload_creates_job_and_status_is_permission_protected(
    monkeypatch,
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

    queued_job_ids: list[int] = []
    monkeypatch.setattr(
        main.ingest_document_task,
        "delay",
        lambda job_id: queued_job_ids.append(job_id),
    )

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

            owner = register("job-owner")
            outsider = register("job-outsider")
            knowledge_base_id = owner["knowledge_bases"][0]["id"]
            owner_headers = {
                "Authorization": f"Bearer {owner['access_token']}",
            }
            outsider_headers = {
                "Authorization": f"Bearer {outsider['access_token']}",
            }

            uploaded = client.post(
                "/documents/upload",
                headers=owner_headers,
                data={"knowledge_base_id": str(knowledge_base_id)},
                files={
                    "file": (
                        "sample.txt",
                        "后台任务测试文本。",
                        "text/plain",
                    )
                },
            )

            assert uploaded.status_code == 202
            job_id = uploaded.json()["id"]
            assert uploaded.json()["status"] == "pending"
            assert queued_job_ids == [job_id]

            forbidden = client.get(
                f"/jobs/{job_id}",
                headers=outsider_headers,
            )
            assert forbidden.status_code == 403

            with session_factory() as session:
                mark_job_processing(session, job_id)
                mark_job_completed(
                    session=session,
                    job_id=job_id,
                    document_id="doc-123",
                    chunking_strategy="heading",
                    chunk_count=3,
                )

            status = client.get(
                f"/jobs/{job_id}",
                headers=owner_headers,
            )
            assert status.status_code == 200
            assert status.json()["status"] == "completed"
            assert status.json()["chunk_count"] == 3
    finally:
        app.dependency_overrides.clear()
