from qdrant_client.models import PointStruct

from enterprise_rag.bm25_retriever import BM25Retriever
from enterprise_rag.hybrid_retriever import HybridRetriever
from enterprise_rag.vector_store import VectorStore


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_hybrid_retriever() -> None:
    store = VectorStore(
        path=":memory:",
        collection_name="hybrid_test",
        vector_size=2,
    )
    store.upsert(
        [
            PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "员工请假需要提前申请"}),
            PointStruct(id=2, vector=[0.0, 1.0], payload={"text": "报销需要提供发票"}),
        ]
    )

    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        bm25_retriever=BM25Retriever(store),
    )

    results = retriever.search("请假申请", limit=1)

    assert results[0].payload["text"] == "员工请假需要提前申请"
    assert results[0].vector_rank == 1
    assert results[0].bm25_rank == 1