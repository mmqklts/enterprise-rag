import json
from types import SimpleNamespace
from typing import Any

from enterprise_rag.agent import (
    AgentModelResponse,
    AgentToolCall,
    KnowledgeAgent,
)
from enterprise_rag.hybrid_retriever import RetrievalResult


class ScriptedAgentModel:
    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AgentModelResponse:
        self.calls.append((messages, tools))
        if not self.responses:
            raise AssertionError("没有剩余脚本化响应")
        return self.responses.pop(0)


class FakeRetriever:
    def search(
        self,
        query: str,
        limit: int = 5,
        knowledge_base_id: str = "default",
    ) -> list[RetrievalResult]:
        del limit, knowledge_base_id
        return [
            RetrievalResult(
                point_id="chunk-1",
                rrf_score=0.5,
                payload={
                    "document_id": "doc-1",
                    "filename": "报销制度.md",
                    "page": 2,
                    "text": f"问题：{query}。报销需要提供发票，并提前提交申请。",
                },
                vector_rank=1,
                bm25_rank=1,
            )
        ]


class FakeReranker:
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        del query
        for result in results:
            result.rerank_score = 0.9
        return results[:limit]


class FakeVectorStore:
    def scroll_all(
        self,
        knowledge_base_id: str = "default",
    ) -> list[Any]:
        del knowledge_base_id
        return [
            SimpleNamespace(
                payload={
                    "document_id": "doc-1",
                    "filename": "报销制度.md",
                    "page": 2,
                }
            ),
            SimpleNamespace(
                payload={
                    "document_id": "doc-2",
                    "filename": "员工手册.md",
                    "page": 1,
                }
            ),
        ]


def tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> AgentModelResponse:
    return AgentModelResponse(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        content=None,
        tool_calls=(
            AgentToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )


def final_answer(content: str) -> AgentModelResponse:
    return AgentModelResponse(
        message={"role": "assistant", "content": content},
        content=content,
        tool_calls=(),
        prompt_tokens=80,
        completion_tokens=30,
        total_tokens=110,
    )


def test_agent_searches_reads_source_and_validates_citation() -> None:
    model = ScriptedAgentModel(
        [
            tool_call(
                "call-1",
                "search_knowledge_base",
                {"query": "报销需要什么材料", "top_k": 3},
            ),
            tool_call(
                "call-2",
                "read_source_chunk",
                {"source_id": 1},
            ),
            final_answer("报销需要提供发票，并提前提交申请 [1]。"),
        ]
    )
    agent = KnowledgeAgent(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        vector_store=FakeVectorStore(),
        model=model,
        max_steps=4,
    )

    result = agent.run("报销需要准备什么？")

    assert [step.tool for step in result.trace] == [
        "search_knowledge_base",
        "read_source_chunk",
    ]
    assert result.trace[0].status == "ok"
    assert result.trace[1].status == "ok"
    assert result.answer == "报销需要提供发票，并提前提交申请 [1]。"
    assert result.sources[0].filename == "报销制度.md"
    assert result.citation_status == "ok"
    assert result.stop_reason == "completed"
    assert result.usage.request_count == 3


def test_agent_forces_final_answer_after_step_budget() -> None:
    model = ScriptedAgentModel(
        [
            tool_call(
                "call-1",
                "search_knowledge_base",
                {"query": "报销流程"},
            ),
            final_answer("应按照制度提交申请 [1]。"),
        ]
    )
    agent = KnowledgeAgent(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        vector_store=FakeVectorStore(),
        model=model,
        max_steps=1,
    )

    result = agent.run("报销流程是什么？")

    assert result.stop_reason == "max_steps"
    assert result.citation_status == "ok"
    assert result.usage.request_count == 2
    assert model.calls[-1][1] is None


def test_agent_skips_duplicate_tool_call() -> None:
    duplicate_arguments = {"query": "请假制度"}
    model = ScriptedAgentModel(
        [
            tool_call(
                "call-1",
                "search_knowledge_base",
                duplicate_arguments,
            ),
            tool_call(
                "call-2",
                "search_knowledge_base",
                duplicate_arguments,
            ),
            final_answer("请假需要提前申请 [1]。"),
        ]
    )
    agent = KnowledgeAgent(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        vector_store=FakeVectorStore(),
        model=model,
        max_steps=4,
    )

    result = agent.run("怎么请假？")

    assert result.trace[1].status == "skipped"
    assert result.citation_status == "ok"


def test_agent_allows_document_listing_without_chunk_sources() -> None:
    model = ScriptedAgentModel(
        [
            tool_call("call-1", "list_documents", {}),
            final_answer("知识库中有《报销制度》和《员工手册》两个文档。"),
        ]
    )
    agent = KnowledgeAgent(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        vector_store=FakeVectorStore(),
        model=model,
        max_steps=3,
    )

    result = agent.run("知识库里有哪些文档？")

    assert result.trace[0].tool == "list_documents"
    assert result.answer == "知识库中有《报销制度》和《员工手册》两个文档。"
    assert result.sources == []
    assert result.citation_status == "not_required"
    assert result.stop_reason == "completed"
