from datetime import UTC, datetime

from sqlalchemy.orm import Session

from enterprise_rag.models import IngestionJob


class InvalidJobStateError(ValueError):
    pass


def create_ingestion_job(
    session: Session,
    knowledge_base_id: int,
    created_by: int,
    filename: str,
    stored_path: str,
    document_id: str | None = None,
) -> IngestionJob:
    job = IngestionJob(
        knowledge_base_id=knowledge_base_id,
        created_by=created_by,
        filename=filename,
        stored_path=stored_path,
        document_id=document_id,
        status="pending",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_ingestion_job(session: Session, job_id: int) -> IngestionJob:
    job = session.get(IngestionJob, job_id)
    if job is None:
        raise LookupError("任务不存在")
    return job


def mark_job_processing(session: Session, job_id: int) -> IngestionJob:
    job = get_ingestion_job(session, job_id)
    if job.status not in {"pending", "failed"}:
        raise InvalidJobStateError("当前任务状态不能开始处理")

    job.status = "processing"
    job.started_at = datetime.now(UTC)
    job.finished_at = None
    job.error_message = None
    job.attempts += 1
    session.commit()
    session.refresh(job)
    return job


def mark_job_completed(
    session: Session,
    job_id: int,
    document_id: str,
    chunking_strategy: str,
    chunk_count: int,
) -> IngestionJob:
    job = get_ingestion_job(session, job_id)
    if job.status != "processing":
        raise InvalidJobStateError("只有处理中的任务可以完成")

    job.status = "completed"
    job.document_id = document_id
    job.chunking_strategy = chunking_strategy
    job.chunk_count = chunk_count
    job.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(job)
    return job


def mark_job_failed(
    session: Session,
    job_id: int,
    error_message: str,
) -> IngestionJob:
    job = get_ingestion_job(session, job_id)
    if job.status not in {"pending", "processing"}:
        raise InvalidJobStateError("只有待处理或处理中的任务可以标记失败")

    job.status = "failed"
    job.error_message = error_message[:2000]
    job.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(job)
    return job
