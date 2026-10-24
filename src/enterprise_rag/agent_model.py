from typing import Any

from enterprise_rag.agent import AgentModelResponse, AgentToolCall
from enterprise_rag.llm_client import DeepSeekClient


class DeepSeekAgentModel:
    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self._client = client or DeepSeekClient()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AgentModelResponse:
        request: dict[str, Any] = {
            "model": self._client.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        response = self._client.client.chat.completions.create(**request)
        message = response.choices[0].message
        tool_calls = tuple(
            AgentToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        )
        usage = response.usage

        return AgentModelResponse(
            message=message.model_dump(exclude_none=True),
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
