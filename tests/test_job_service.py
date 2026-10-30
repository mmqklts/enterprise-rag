import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag.auth_service import register_user
from enterprise_rag.db import Base
from enterprise_rag.job_service import (
    InvalidJobStateError,
    create_ingestion_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_processing,
)


def create_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_ingestion_job_state_transitions() -> None:
    with create_session() as session:
        account = register_user(
            session=session,
            username="job-user",
            email="job-user@example.com",
            password="strong-password",
        )
        job = create_ingestion_job(
            session=session,
            knowledge_base_id=account.knowledge_base.id,
            created_by=account.user.id,
            filename="sample.md",
            stored_path="data/uploads/sample.md",
        )

        assert job.status == "pending"
        assert job.attempts == 0

        processing = mark_job_processing(session, job.id)
        assert processing.status == "processing"
        assert processing.attempts == 1
        assert processing.started_at is not None

        completed = mark_job_completed(
            session=session,
            job_id=job.id,
            document_id="doc-123",
            chunking_strategy="heading",
            chunk_count=4,
        )
        assert completed.status == "completed"
        assert completed.chunk_count == 4
        assert completed.finished_at is not None

        with pytest.raises(InvalidJobStateError):
            mark_job_processing(session, job.id)

        failed_job = create_ingestion_job(
            session=session,
            knowledge_base_id=account.knowledge_base.id,
            created_by=account.user.id,
            filename="broken.pdf",
            stored_path="data/uploads/broken.pdf",
        )
        mark_job_processing(session, failed_job.id)
        failed = mark_job_failed(
            session=session,
            job_id=failed_job.id,
            error_message="PDF 解析失败",
        )

        assert failed.status == "failed"
        assert failed.error_message == "PDF 解析失败"
        assert failed.finished_at is not None