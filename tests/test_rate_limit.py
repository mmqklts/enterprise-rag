import pytest
from fastapi import HTTPException
from starlette.requests import Request

from enterprise_rag import rate_limit


class FakeRedis:
    def __init__(self, current: int, ttl: int) -> None:
        self.current = current
        self.ttl = ttl

    def eval(self, *args):
        del args
        return [self.current, self.ttl]


def make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )


def test_rate_limit_rejects_requests_over_limit(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setattr(
        rate_limit,
        "get_redis_client",
        lambda: FakeRedis(current=2, ttl=42),
    )

    with pytest.raises(HTTPException) as exc_info:
        rate_limit.enforce_rate_limit(
            make_request(),
            bucket="chat",
            limit=1,
            window_seconds=60,
            subject="user:7",
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["Retry-After"] == "42"
    assert exc_info.value.headers["X-RateLimit-Limit"] == "1"


def test_rate_limit_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    rate_limit.enforce_rate_limit(
        make_request(),
        bucket="chat",
        limit=1,
        window_seconds=60,
    )
