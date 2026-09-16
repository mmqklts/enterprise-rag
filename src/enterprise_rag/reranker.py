import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from enterprise_rag.hybrid_retriever import RetrievalResult


class Reranker:
    def __init__(
        self,
        model_path: str | Path | None = None,
        model: CrossEncoder | None = None,
    ) -> None:
        if model is not None:
            self.model = model
            return

        load_dotenv()
        configured_path = model_path or os.getenv("RERANKER_MODEL_PATH")

        if not configured_path:
            raise RuntimeError("RERANKER_MODEL_PATH 未配置")

        self.model = CrossEncoder(str(configured_path))

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [
            (query, str(result.payload.get("text", "")))
            for result in results
        ]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores, strict=True):
            result.rerank_score = float(score)

        return sorted(
            results,
            key=lambda result: result.rerank_score or float("-inf"),
            reverse=True,
        )[:limit]