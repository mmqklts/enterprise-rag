import os

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        return vectors.tolist()