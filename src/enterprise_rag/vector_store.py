from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, Record, ScoredPoint, VectorParams


class VectorStore:
    def __init__(
        self,
        path: str | Path = "data/qdrant",
        collection_name: str = "knowledge_chunks",
        vector_size: int = 512,
    ) -> None:
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

    def search(self, vector: list[float], limit: int = 5) -> list[ScoredPoint]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return result.points
    
    def scroll_all(self) -> list[Record]:
        records: list[Record] = []
        offset = None

        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(batch)

            if offset is None:
                break

        return records