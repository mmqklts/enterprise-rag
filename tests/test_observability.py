import importlib

from fastapi.testclient import TestClient

from enterprise_rag.observability import estimate_llm_cost

main = importlib.import_module("enterprise_rag.main")


def test_request_id_is_returned() -> None:
    with TestClient(main.app) as client:
        response = client.get(
            "/health",
            headers={"X-Request-ID": "test-request-id"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_estimate_llm_cost(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CACHE_HIT_PRICE_PER_MILLION", "0.04")
    monkeypatch.setenv("LLM_CACHE_MISS_PRICE_PER_MILLION", "2")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_MILLION", "8")

    cost = estimate_llm_cost(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_hit_tokens=400,
        cache_miss_tokens=600,
    )

    assert round(cost, 6) == 0.002816
