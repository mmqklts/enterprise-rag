from dataclasses import dataclass
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from enterprise_rag.vector_store import VectorStore


@dataclass
class BM25Result:
    point_id: str | int
    score: float
    payload: dict[str, Any]


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in jieba.lcut(text)
        if token.strip() and token.isalnum()
    ]


class BM25Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def search(self, query: str, limit: int = 5) -> list[BM25Result]:
        records = self.vector_store.scroll_all()
        if not records:
            return []

        tokenized_texts = [
            tokenize(str((record.payload or {}).get("text", "")))
            for record in records
        ]

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized_texts)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            BM25Result(
                point_id=records[index].id,
                score=float(score),
                payload=records[index].payload or {},
            )
            for index, score in ranked[:limit]
        ]