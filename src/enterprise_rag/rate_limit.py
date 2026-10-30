import hashlib
import logging
import os
from functools import lru_cache

from fastapi import HTTPException, Request, status
from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

_RATE_LIMIT_SCRIPT = """
local current = redis.call("INCR", KEYS[1])
if current == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
local ttl = redis.call("TTL", KEYS[1])
return {current, ttl}
"""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def rate_limit_enabled() -> bool:
    production = os.getenv("APP_ENV", "development").lower() == "production"
    return _env_bool("RATE_LIMIT_ENABLED", production)


@lru_cache
def get_redis_client() -> Redis:
    redis_url = os.getenv(
        "RATE_LIMIT_REDIS_URL",
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )
    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )


def _request_identity(request: Request, subject: str | None) -> str:
    if subject:
        return subject
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def enforce_rate_limit(
    request: Request,
    *,
    bucket: str,
    limit: int,
    window_seconds: int,
    subject: str | None = None,
) -> None:
    if not rate_limit_enabled():
        return

    identity = _request_identity(request, subject)
    identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    key = f"rate-limit:{bucket}:{identity_digest}"

    try:
        current, ttl = get_redis_client().eval(
            _RATE_LIMIT_SCRIPT,
            1,
            key,
            window_seconds,
        )
    except RedisError as exc:
        production = os.getenv("APP_ENV", "development").lower() == "production"
        fail_open = _env_bool("RATE_LIMIT_FAIL_OPEN", not production)
        if fail_open:
            logger.warning("rate_limit_backend_unavailable", exc_info=exc)
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="限流服务暂时不可用",
        ) from exc

    if int(current) > limit:
        retry_after = max(1, int(ttl))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后重试",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
            },
        )
