from qdrant_client.models import PointStruct

from enterprise_rag.vector_store import VectorStore


def test_vector_store_search() -> None:
    store = VectorStore(
        path=":memory:",
        collection_name="test_chunks",
        vector_size=2,
    )

    store.upsert(
        [
            PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "请假"}),
            PointStruct(id=2, vector=[0.0, 1.0], payload={"text": "报销"}),
        ]
    )

    results = store.search([1.0, 0.0], limit=1)

    assert results[0].id == 1
    assert results[0].payload["text"] == "请假"


def test_vector_store_deletes_document_chunks() -> None:
    store = VectorStore(
        path=":memory:",
        collection_name="delete_test_chunks",
        vector_size=2,
    )
    store.upsert(
        [
            PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"document_id": "doc-1", "text": "旧块一"},
            ),
            PointStruct(
                id=2,
                vector=[0.5, 0.5],
                payload={"document_id": "doc-1", "text": "旧块二"},
            ),
            PointStruct(
                id=3,
                vector=[0.0, 1.0],
                payload={"document_id": "doc-2", "text": "其他文档"},
            ),
        ]
    )

    store.delete_by_document_id("doc-1")

    remaining = store.scroll_all()
    assert len(remaining) == 1
    assert remaining[0].payload["document_id"] == "doc-2"
