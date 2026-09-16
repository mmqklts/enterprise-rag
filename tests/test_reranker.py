from enterprise_rag.hybrid_retriever import RetrievalResult
from enterprise_rag.reranker import Reranker


class FakeModel:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.2, 0.9]


def test_reranker() -> None:
    results = [
        RetrievalResult(point_id=1, rrf_score=0.1, payload={"text": "请假"}, vector_rank=1),
        RetrievalResult(point_id=2, rrf_score=0.2, payload={"text": "报销"}, vector_rank=2),
    ]

    reranker = Reranker(model=FakeModel())
    reranked = reranker.rerank("问题", results, limit=1)

    assert reranked[0].point_id == 2
    assert reranked[0].rerank_score == 0.9