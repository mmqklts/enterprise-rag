import importlib
from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_rag.db import Base, get_db
from enterprise_rag.rag import RagMetrics, RagResponse

main = importlib.import_module("enterprise_rag.main")


class FakeRagService:
    def answer(
        self,
        question: str,
        limit: int,
        knowledge_base_id: str,
    ) -> RagResponse:
        del limit
        return RagResponse(
            answer=f"{knowledge_base_id}:{question}",
            sources=[],
            metrics=RagMetrics(
                retrieval_ms=1,
                rerank_ms=1,
                llm_ms=1,
                total_ms=3,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cache_hit_tokens=0,
                cache_miss_tokens=1,
            ),
        )


def test_chat_requires_auth_and_knowledge_base_membership(
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

    app = main.app
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main, "get_rag_service", FakeRagService)

    try:
        with TestClient(app) as client:
            unauthorized = client.post(
                "/chat",
                json={
                    "question": "请假多久？",
                    "knowledge_base_id": 1,
                },
            )
            assert unauthorized.status_code == 401

            registered = client.post(
                "/auth/register",
                json={
                    "username": "api-user",
                    "email": "api-user@example.com",
                    "password": "strong-password",
                },
            )
            assert registered.status_code == 200
            auth = registered.json()
            knowledge_base_id = auth["knowledge_bases"][0]["id"]
            headers = {
                "Authorization": f"Bearer {auth['access_token']}",
            }

            allowed = client.post(
                "/chat",
                headers=headers,
                json={
                    "question": "请假多久？",
                    "knowledge_base_id": knowledge_base_id,
                },
            )
            assert allowed.status_code == 200
            assert allowed.json()["answer"].startswith(
                f"{knowledge_base_id}:"
            )

            forbidden = client.post(
                "/chat",
                headers=headers,
                json={
                    "question": "请假多久？",
                    "knowledge_base_id": 999,
                },
            )
            assert forbidden.status_code == 403
    finally:
        app.dependency_overrides.clear()
