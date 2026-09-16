from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct

from enterprise_rag.chunker import chunk_text
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.pdf_reader import read_pdf_pages
from enterprise_rag.text_reader import read_text_file
from enterprise_rag.vector_store import VectorStore


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentIngestionService:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
        filename: str,
    ) -> IngestionResult:
        path = Path(file_path)
        pages = self._read_pages(path)

        chunk_texts: list[str] = []
        payloads: list[dict[str, str | int]] = []

        for page_number, page_text in pages:
            for chunk in chunk_text(
                page_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            ):
                chunk_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{document_id}:{page_number}:{chunk.chunk_index}",
                    )
                )

                chunk_texts.append(chunk.text)
                payloads.append(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "page": page_number,
                        "chunk_index": chunk.chunk_index,
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

        self.vector_store.upsert(points)

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            chunk_count=len(points),
        )

    def _read_pages(self, path: Path) -> list[tuple[int, str]]:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return [
                (page.page_number, page.text)
                for page in read_pdf_pages(path)
            ]

        if suffix in {".md", ".markdown", ".txt"}:
            return [(1, read_text_file(path))]

        raise ValueError(f"不支持的文件类型：{suffix}")