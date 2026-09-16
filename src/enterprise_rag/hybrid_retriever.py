from dataclasses import dataclass
from typing import Any

from enterprise_rag.bm25_retriever import BM25Retriever
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.vector_store import VectorStore


@dataclass
class RetrievalResult:
    point_id: str | int
    rrf_score: float
    payload: dict[str, Any]
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

class HybridRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        query_vector = self.embedder.embed_texts([query])[0]
        vector_results = self.vector_store.search(query_vector, limit=limit * 2)
        bm25_results = self.bm25_retriever.search(query, limit=limit * 2)

        candidates: dict[str | int, RetrievalResult] = {}

        for rank, point in enumerate(vector_results, start=1):
            candidates[point.id] = RetrievalResult(
                point_id=point.id,
                rrf_score=1 / (self.rrf_k + rank),
                payload=point.payload or {},
                vector_rank=rank,
            )

        for rank, result in enumerate(bm25_results, start=1):
            candidate = candidates.get(result.point_id)

            if candidate is None:
                candidates[result.point_id] = RetrievalResult(
                    point_id=result.point_id,
                    rrf_score=1 / (self.rrf_k + rank),
                    payload=result.payload,
                    bm25_rank=rank,
                )
            else:
                candidate.rrf_score += 1 / (self.rrf_k + rank)
                candidate.bm25_rank = rank

        return sorted(
            candidates.values(),
            key=lambda result: result.rrf_score,
            reverse=True,
        )[:limit]