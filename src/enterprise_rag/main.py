from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.ingestion import DocumentIngestionService, IngestionResult
from enterprise_rag.llm_client import DeepSeekClient
from enterprise_rag.rag import RagResponse, RagService
from enterprise_rag.vector_store import VectorStore
from enterprise_rag.bm25_retriever import BM25Retriever
from enterprise_rag.hybrid_retriever import HybridRetriever
from enterprise_rag.reranker import Reranker

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from hashlib import sha256
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
from dotenv import load_dotenv

load_dotenv()

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)

UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_SIZE = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

app = FastAPI(title="Enterprise RAG API")
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

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=RagResponse)
def chat(request: ChatRequest) -> RagResponse:
    rag_service = get_rag_service()
    return rag_service.answer(
        question=request.question,
        limit=request.limit,
    )

@app.post("/documents/upload", response_model=IngestionResult)
async def upload_document(
    file: UploadFile = File(...),
) -> IngestionResult:
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

    try:
        result = get_ingestion_service().ingest(
            file_path=destination,
            document_id=document_id,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result