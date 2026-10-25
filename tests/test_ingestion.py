from typing import Any

from qdrant_client.models import PointStruct

from enterprise_rag.ingestion import DocumentIngestionService


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: list[PointStruct] = []
        self.deleted_document_ids: list[str] = []

    def delete_by_document_id(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)

    def upsert(self, points: list[PointStruct]) -> None:
        self.points.extend(points)


def test_ingestion_uses_detected_heading_structure(tmp_path: Any) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """
# 公司制度

## 请假制度
员工请假需要提前申请。

## 报销制度
报销需要提供发票。

## 安全制度
密码不得少于十二位。
""",
        encoding="utf-8",
    )
    store = FakeVectorStore()
    service = DocumentIngestionService(
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    result = service.ingest(
        file_path=path,
        document_id="doc-1",
        filename="policy.md",
    )

    assert result.chunking_strategy == "heading"
    assert result.chunk_count == len(store.points)
    assert store.deleted_document_ids == ["doc-1"]
    payloads = [point.payload or {} for point in store.points]
    assert any(
        payload["section_title"] == "请假制度"
        for payload in payloads
    )
    assert all(
        payload["chunking_strategy"] == "heading"
        for payload in payloads
    )
