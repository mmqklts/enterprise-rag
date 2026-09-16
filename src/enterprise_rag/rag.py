from dataclasses import dataclass

from enterprise_rag.hybrid_retriever import HybridRetriever
from enterprise_rag.llm_client import DeepSeekClient
from enterprise_rag.reranker import Reranker


@dataclass
class Source:
    index: int
    document_id: str
    filename: str
    text: str
    page: int | None
    score: float
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None


@dataclass
class RagResponse:
    answer: str
    sources: list[Source]


class RagService:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: DeepSeekClient,
        reranker: Reranker,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def answer(self, question: str, limit: int = 5) -> RagResponse:
        results = self.retriever.search(
            question, 
            limit=max(limit * 4, 10),
        )
        results = self.reranker.rerank(
            question,
            results,
            limit=limit,
        )
        if not results:
            return RagResponse(answer="知识库中没有找到相关内容。", sources=[])

        sources: list[Source] = []
        context_parts: list[str] = []

        for index, result in enumerate(results, start=1):
            payload = result.payload or {}
            document_id = str(payload.get("document_id", ""))
            filename = str(payload.get("filename", ""))
            text = str(payload.get("text", ""))
            raw_page = payload.get("page")
            page = raw_page if isinstance(raw_page, int) else None
            sources.append(
                Source(
                    index=index,
                    document_id=document_id,
                    filename=filename,
                    text=text,
                    page=page,
                    score=(
                        result.rerank_score
                        if result.rerank_score is not None
                        else result.rrf_score
                    ),
                    rrf_score=result.rrf_score,
                    vector_rank=result.vector_rank,
                    bm25_rank=result.bm25_rank,
                    rerank_score=result.rerank_score,
                )
            )

            page_label = f"{filename} 第 {page} 页" if page is not None else filename
            context_parts.append(f"[{index}] 来源：{page_label}\n{text}")

        prompt = f"""
你是企业知识库问答助手。
只能根据参考资料回答，不能自行编造。
每个关键结论必须使用 [1]、[2] 这样的编号标注来源。
如果资料不足，请明确回答“知识库中的资料不足以回答该问题”。

参考资料：
{chr(10).join(context_parts)}

用户问题：{question}
"""

        answer = self.llm.generate(prompt)
        return RagResponse(answer=answer, sources=sources)