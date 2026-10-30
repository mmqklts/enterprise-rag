from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from enterprise_rag.auth_routes import get_current_user
from enterprise_rag.db import get_db
from enterprise_rag.job_service import get_ingestion_job
from enterprise_rag.models import IngestionJob, User
from enterprise_rag.permissions import require_knowledge_base_access

router = APIRouter(prefix="/jobs", tags=["jobs"])


class IngestionJobResponse(BaseModel):
    id: int
    knowledge_base_id: int
    created_by: int
    document_id: str | None
    filename: str
    status: str
    chunking_strategy: str | None
    chunk_count: int
    attempts: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        knowledge_base_id=job.knowledge_base_id,
        created_by=job.created_by,
        document_id=job.document_id,
        filename=job.filename,
        status=job.status,
        chunking_strategy=job.chunking_strategy,
        chunk_count=job.chunk_count,
        attempts=job.attempts,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/{job_id}", response_model=IngestionJobResponse)
def get_job(
    job_id: int,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> IngestionJobResponse:
    try:
        job = get_ingestion_job(session, job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=job.knowledge_base_id,
        allowed_roles={"owner", "editor", "viewer"},
    )
    return job_response(job)
