import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from enterprise_rag.bm25_retriever import BM25Retriever
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.hybrid_retriever import HybridRetriever
from enterprise_rag.ingestion import DocumentIngestionService
from enterprise_rag.reranker import Reranker
from enterprise_rag.vector_store import VectorStore


@dataclass
class EvalQuestion:
    question: str
    expected_text: str


@dataclass
class MetricSummary:
    mode: str
    recall_at_1: float
    recall_at_5: float
    mrr: float


class RetrievalEvaluator:
    def __init__(self, corpus_dir: Path) -> None:
        self.embedder = TextEmbedder()
        self.store = VectorStore(
            path=":memory:",
            collection_name="evaluation_chunks",
            vector_size=512,
        )
        self.ingestion = DocumentIngestionService(
            embedder=self.embedder,
            vector_store=self.store,
        )
        self.bm25 = BM25Retriever(self.store)
        self.hybrid = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.store,
            bm25_retriever=self.bm25,
        )
        self.reranker = Reranker()

        self._ingest_corpus(corpus_dir)

    def _ingest_corpus(self, corpus_dir: Path) -> None:
        for path in sorted(corpus_dir.iterdir()):
            if path.suffix.lower() not in {".pdf", ".md", ".markdown", ".txt"}:
                continue

            document_id = sha256(path.read_bytes()).hexdigest()
            self.ingestion.ingest(
                file_path=path,
                document_id=document_id,
                filename=path.name,
            )

    def _payloads(
        self,
        mode: str,
        question: str,
        limit: int = 5,
    ) -> list[dict]:
        if mode == "vector":
            vector = self.embedder.embed_texts([question])[0]
            points = self.store.search(vector, limit=limit)
            return [point.payload or {} for point in points]

        if mode == "bm25":
            results = self.bm25.search(question, limit=limit)
            return [result.payload for result in results]

        if mode == "hybrid":
            results = self.hybrid.search(question, limit=limit)
            return [result.payload for result in results]

        if mode == "rerank":
            candidates = self.hybrid.search(question, limit=20)
            results = self.reranker.rerank(question, candidates, limit=limit)
            return [result.payload for result in results]

        raise ValueError(f"未知检索模式：{mode}")

    def evaluate_mode(
        self,
        mode: str,
        questions: list[EvalQuestion],
    ) -> MetricSummary:
        recall_at_1 = 0
        recall_at_5 = 0
        reciprocal_rank_sum = 0.0

        for item in questions:
            payloads = self._payloads(mode, item.question)
            rank = None

            for index, payload in enumerate(payloads, start=1):
                text = str(payload.get("text", ""))
                if item.expected_text in text:
                    rank = index
                    break

            if rank is None:
                continue

            if rank == 1:
                recall_at_1 += 1

            if rank <= 5:
                recall_at_5 += 1

            reciprocal_rank_sum += 1 / rank

        count = len(questions)
        return MetricSummary(
            mode=mode,
            recall_at_1=recall_at_1 / count,
            recall_at_5=recall_at_5 / count,
            mrr=reciprocal_rank_sum / count,
        )

    def evaluate_all(
        self,
        questions: list[EvalQuestion],
    ) -> list[MetricSummary]:
        return [
            self.evaluate_mode(mode, questions)
            for mode in ("vector", "bm25", "hybrid", "rerank")
        ]


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            questions.append(
                EvalQuestion(
                    question=data["question"],
                    expected_text=data["expected_text"],
                )
            )

    return questions


def main() -> None:
    questions = load_questions(Path("data/eval/questions.jsonl"))
    evaluator = RetrievalEvaluator(Path("data/eval/corpus"))
    summaries = evaluator.evaluate_all(questions)

    print(f"{'模式':<10} {'Recall@1':>10} {'Recall@5':>10} {'MRR':>10}")
    for summary in summaries:
        print(
            f"{summary.mode:<10} "
            f"{summary.recall_at_1:>10.3f} "
            f"{summary.recall_at_5:>10.3f} "
            f"{summary.mrr:>10.3f}"
        )


if __name__ == "__main__":
    main()