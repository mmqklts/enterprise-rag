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