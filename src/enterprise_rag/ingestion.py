from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct

from enterprise_rag.chunking_router import ChunkingRouter
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.pdf_reader import PdfPage, read_pdf_pages
from enterprise_rag.structure_detector import ChunkingStrategy
from enterprise_rag.text_reader import read_text_file
from enterprise_rag.vector_store import VectorStore


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    chunking_strategy: str
    structure_confidence: float


class DocumentIngestionService:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        chunk_size: int = 500,
        overlap: int = 100,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunking_strategy = chunking_strategy
        self.router = ChunkingRouter(
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
        filename: str,
    ) -> IngestionResult:
        path = Path(file_path)
        pages = self._read_pages(path)
        chunking = self.router.route(
            pages,
            strategy=self.chunking_strategy,
        )

        chunk_texts: list[str] = []
        payloads: list[dict[str, str | int]] = []

        for chunk in chunking.chunks:
            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{document_id}:{chunk.strategy.value}:"
                        f"{chunk.page_start}:{chunk.chunk_index}"
                    ),
                )
            )

            chunk_texts.append(chunk.text)
            payloads.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page": chunk.page_start,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.chunk_index,
                    "chunking_strategy": chunk.strategy.value,
                    "structure_confidence": chunking.profile.confidence,
                    "section_title": chunk.section_title or "",
                    "article_number": chunk.article_number or "",
                    "article_title": chunk.article_title or "",
                    "text": chunk.text,
                }
            )

        if not chunk_texts:
            raise ValueError("文档中没有可索引的文字")

        vectors = self.embedder.embed_texts(chunk_texts)

        points = [
            PointStruct(
                id=payload["chunk_id"],
                vector=vector,
                payload=payload,
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]

        self.vector_store.delete_by_document_id(document_id)
        self.vector_store.upsert(points)

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            chunk_count=len(points),
            chunking_strategy=chunking.strategy.value,
            structure_confidence=chunking.profile.confidence,
        )

    def _read_pages(self, path: Path) -> list[PdfPage]:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return read_pdf_pages(path)

        if suffix in {".md", ".markdown", ".txt"}:
            return [
                PdfPage(
                    page_number=1,
                    text=read_text_file(path),
                )
            ]

        raise ValueError(f"不支持的文件类型：{suffix}")
