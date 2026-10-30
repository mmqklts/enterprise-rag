import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from enterprise_rag.celery_app import celery_app
from enterprise_rag.db import SessionLocal
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.ingestion import DocumentIngestionService
from enterprise_rag.job_service import (
    get_ingestion_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_processing,
)
from enterprise_rag.vector_store import VectorStore


@lru_cache
def get_worker_embedder() -> TextEmbedder:
    return TextEmbedder()


@lru_cache
def get_worker_vector_store() -> VectorStore:
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        return VectorStore(url=qdrant_url)

    return VectorStore()


@celery_app.task(bind=True, max_retries=2)
def ingest_document_task(self, job_id: int) -> dict[str, str | int]:
    session = SessionLocal()

    try:
        job = get_ingestion_job(session, job_id)
        mark_job_processing(session, job_id)

        path = Path(job.stored_path)
        document_id = job.document_id or sha256(path.read_bytes()).hexdigest()
        service = DocumentIngestionService(
            embedder=get_worker_embedder(),
            vector_store=get_worker_vector_store(),
        )
        result = service.ingest(
            file_path=path,
            document_id=document_id,
            filename=job.filename,
            knowledge_base_id=str(job.knowledge_base_id),
        )
        mark_job_completed(
            session=session,
            job_id=job_id,
            document_id=result.document_id,
            chunking_strategy=result.chunking_strategy,
            chunk_count=result.chunk_count,
        )
        return {
            "job_id": job_id,
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
        }
    except Exception as exc:
        try:
            job = get_ingestion_job(session, job_id)
            if job.status in {"pending", "processing"}:
                mark_job_failed(session, job_id, str(exc))
        finally:
            session.close()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=5) from exc
        raise

    finally:
        session.close()
