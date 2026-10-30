import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from enterprise_rag.agent import KnowledgeAgent
from enterprise_rag.agent_model import DeepSeekAgentModel
from enterprise_rag.auth_routes import get_current_user
from enterprise_rag.auth_routes import router as auth_router
from enterprise_rag.bm25_retriever import BM25Retriever
from enterprise_rag.db import get_db
from enterprise_rag.document_service import (
    delete_document_record,
    get_document_record,
    list_document_records,
)
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.hybrid_retriever import HybridRetriever
from enterprise_rag.ingestion import DocumentIngestionService
from enterprise_rag.job_routes import (
    IngestionJobResponse,
    job_response,
)
from enterprise_rag.job_routes import (
    router as job_router,
)
from enterprise_rag.job_service import (
    create_ingestion_job,
    mark_job_failed,
)
from enterprise_rag.knowledge_base_routes import (
    router as knowledge_base_router,
)
from enterprise_rag.llm_client import DeepSeekClient
from enterprise_rag.models import User
from enterprise_rag.observability import (
    RequestLoggingMiddleware,
    estimate_llm_cost,
)
from enterprise_rag.observability import (
    logger as request_logger,
)
from enterprise_rag.permissions import require_knowledge_base_access
from enterprise_rag.rag import RagResponse, RagService
from enterprise_rag.reranker import Reranker
from enterprise_rag.tasks import ingest_document_task
from enterprise_rag.vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    knowledge_base_id: int = Field(gt=0)


class AgentSourceResponse(BaseModel):
    index: int
    document_id: str
    filename: str
    text: str
    page: int | None
    score: float
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None


class AgentStepResponse(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any]
    status: str
    summary: str
    duration_ms: float


class AgentUsageResponse(BaseModel):
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AgentChatResponse(BaseModel):
    answer: str
    sources: list[AgentSourceResponse]
    trace: list[AgentStepResponse]
    usage: AgentUsageResponse
    stop_reason: str
    citation_status: str
    duration_ms: float


class DocumentResponse(BaseModel):
    knowledge_base_id: int
    document_id: str
    filename: str
    chunking_strategy: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_SIZE = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

app = FastAPI(title="Enterprise Knowledge Agent API")
app.add_middleware(RequestLoggingMiddleware)
app.include_router(auth_router)
app.include_router(knowledge_base_router)
app.include_router(job_router)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@lru_cache
def get_embedder() -> TextEmbedder:
    return TextEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        return VectorStore(url=qdrant_url)

    return VectorStore()


@lru_cache
def get_llm() -> DeepSeekClient:
    return DeepSeekClient()


@lru_cache
def get_rag_service() -> RagService:
    return RagService(
        retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        llm=get_llm(),
    )


@lru_cache
def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever(get_vector_store())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        bm25_retriever=get_bm25_retriever(),
    )


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache
def get_agent() -> KnowledgeAgent:
    configured_steps = int(os.getenv("AGENT_MAX_STEPS", "6"))

    return KnowledgeAgent(
        retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        vector_store=get_vector_store(),
        model=DeepSeekAgentModel(),
        max_steps=max(1, min(configured_steps, 12)),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=RagResponse)
def chat(
    payload: ChatRequest,
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> RagResponse:
    knowledge_base = require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=payload.knowledge_base_id,
        allowed_roles={"owner", "editor", "viewer"},
    )
    rag_service = get_rag_service()
    result = rag_service.answer(
        question=payload.question,
        limit=payload.limit,
        knowledge_base_id=str(knowledge_base.id),
    )
    request_logger.info(
        json.dumps(
            {
                "event": "rag_request",
                "request_id": http_request.state.request_id,
                "user_id": user.id,
                "knowledge_base_id": knowledge_base.id,
                "retrieval_ms": result.metrics.retrieval_ms,
                "rerank_ms": result.metrics.rerank_ms,
                "llm_ms": result.metrics.llm_ms,
                "total_ms": result.metrics.total_ms,
                "prompt_tokens": result.metrics.prompt_tokens,
                "completion_tokens": result.metrics.completion_tokens,
                "total_tokens": result.metrics.total_tokens,
                "estimated_cost": estimate_llm_cost(
                    prompt_tokens=result.metrics.prompt_tokens,
                    completion_tokens=result.metrics.completion_tokens,
                    cache_hit_tokens=result.metrics.cache_hit_tokens,
                    cache_miss_tokens=result.metrics.cache_miss_tokens,
                ),
            },
            ensure_ascii=False,
        )
    )
    return result


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(
    payload: ChatRequest,
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> AgentChatResponse:
    knowledge_base = require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=payload.knowledge_base_id,
        allowed_roles={"owner", "editor", "viewer"},
    )
    try:
        result = get_agent().run(
            question=payload.question,
            limit=payload.limit,
            knowledge_base_id=str(knowledge_base.id),
        )
    except Exception as exc:
        logger.exception("agent_request_failed")
        raise HTTPException(
            status_code=503,
            detail="Agent 暂时不可用，请检查模型配置或查看服务日志。",
        ) from exc

    request_logger.info(
        json.dumps(
            {
                "event": "agent_request",
                "request_id": http_request.state.request_id,
                "user_id": user.id,
                "knowledge_base_id": knowledge_base.id,
                "duration_ms": result.duration_ms,
                "request_count": result.usage.request_count,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "estimated_cost": estimate_llm_cost(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                ),
                "stop_reason": result.stop_reason,
            },
            ensure_ascii=False,
        )
    )

    return AgentChatResponse(
        answer=result.answer,
        sources=[
            AgentSourceResponse(
                index=source.index,
                document_id=source.document_id,
                filename=source.filename,
                text=source.text,
                page=source.page,
                score=source.score,
                rrf_score=source.rrf_score,
                vector_rank=source.vector_rank,
                bm25_rank=source.bm25_rank,
                rerank_score=source.rerank_score,
            )
            for source in result.sources
        ],
        trace=[
            AgentStepResponse(
                step=step.step,
                tool=step.tool,
                arguments=step.arguments,
                status=step.status,
                summary=step.summary,
                duration_ms=step.duration_ms,
            )
            for step in result.trace
        ],
        usage=AgentUsageResponse(
            request_count=result.usage.request_count,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        stop_reason=result.stop_reason,
        citation_status=result.citation_status,
        duration_ms=result.duration_ms,
    )


@app.post(
    "/documents/upload",
    response_model=IngestionJobResponse,
    status_code=202,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    knowledge_base_id: Annotated[int, Form(gt=0)],
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> IngestionJobResponse:
    knowledge_base = require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner", "editor"},
    )
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="仅支持 PDF、Markdown 和 TXT")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="文件不能超过 20MB")

    document_id = sha256(content).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / f"{document_id}{suffix}"
    destination.write_bytes(content)

    job = create_ingestion_job(
        session=session,
        knowledge_base_id=knowledge_base.id,
        created_by=user.id,
        filename=filename,
        stored_path=str(destination),
        document_id=document_id,
    )

    try:
        ingest_document_task.delay(job.id)
    except Exception as exc:
        mark_job_failed(session, job.id, str(exc))
        raise HTTPException(
            status_code=503,
            detail="任务队列暂时不可用",
        ) from exc

    return job_response(job)


@app.get(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=list[DocumentResponse],
)
def list_documents(
    knowledge_base_id: int,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> list[DocumentResponse]:
    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner", "editor", "viewer"},
    )
    records = list_document_records(session, knowledge_base_id)
    return [
        DocumentResponse(
            knowledge_base_id=record.knowledge_base_id,
            document_id=record.document_id,
            filename=record.filename,
            chunking_strategy=record.chunking_strategy,
            chunk_count=record.chunk_count,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        for record in records
    ]


@app.delete(
    "/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
    status_code=204,
)
def delete_document(
    knowledge_base_id: int,
    document_id: str,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> None:
    require_knowledge_base_access(
        session=session,
        user_id=user.id,
        knowledge_base_id=knowledge_base_id,
        allowed_roles={"owner", "editor"},
    )

    try:
        record = get_document_record(
            session=session,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    get_vector_store().delete_by_document_id(
        document_id,
        knowledge_base_id=str(knowledge_base_id),
    )

    stored_path = Path(record.stored_path).resolve()
    upload_root = UPLOAD_DIR.resolve()
    if upload_root in stored_path.parents and stored_path.exists():
        stored_path.unlink()

    delete_document_record(session, record)
