from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)


class VectorStore:
    def __init__(
        self,
        path: str | Path = "data/qdrant",
        collection_name: str = "knowledge_chunks",
        vector_size: int = 512,
        url: str | None = None,
    ) -> None:
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=str(path))

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, points: list[PointStruct]) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def delete_by_document_id(
        self,
        document_id: str,
        knowledge_base_id: str = "default",
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                    FieldCondition(
                        key="knowledge_base_id",
                        match=MatchValue(value=knowledge_base_id),
                    ),
                ]
            ),
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        knowledge_base_id: str = "default",
    ) -> list[ScoredPoint]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="knowledge_base_id",
                        match=MatchValue(value=knowledge_base_id),
                    )
                ]
            ),
            with_payload=True,
        )
        return result.points

    def scroll_all(
        self,
        knowledge_base_id: str = "default",
    ) -> list[Record]:
        records: list[Record] = []
        offset = None

        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="knowledge_base_id",
                            match=MatchValue(value=knowledge_base_id),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )
            records.extend(batch)

            if offset is None:
                break

        return records
