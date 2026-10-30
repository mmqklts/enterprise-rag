import json
import logging
import os
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("enterprise_rag.requests")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

logger.propagate = False

audit_logger = logging.getLogger("enterprise_rag.audit")
audit_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

if not audit_logger.handlers:
    audit_handler = logging.StreamHandler()
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(audit_handler)

audit_logger.propagate = False


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": 500,
                        "duration_ms": duration_ms,
                    },
                    ensure_ascii=False,
                )
            )
            raise

        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if (
            request.url.scheme == "https"
            or forwarded_proto.split(",", maxsplit=1)[0].strip() == "https"
        ):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
                ensure_ascii=False,
            )
        )
        return response


def audit_log(
    *,
    event: str,
    action: str,
    outcome: str,
    request_id: str,
    actor_user_id: int | None = None,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    **details: object,
) -> None:
    audit_logger.info(
        json.dumps(
            {
                "event": event,
                "action": action,
                "outcome": outcome,
                "request_id": request_id,
                "actor_user_id": actor_user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                **details,
            },
            ensure_ascii=False,
        )
    )


def estimate_llm_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int | None = None,
) -> float:
    cache_hit_price = float(
        os.getenv("LLM_CACHE_HIT_PRICE_PER_MILLION", "0")
    )
    cache_miss_price = float(
        os.getenv("LLM_CACHE_MISS_PRICE_PER_MILLION", "0")
    )
    output_price = float(
        os.getenv("LLM_OUTPUT_PRICE_PER_MILLION", "0")
    )
    miss_tokens = (
        cache_miss_tokens
        if cache_miss_tokens is not None
        else prompt_tokens
    )

    return (
        cache_hit_tokens * cache_hit_price
        + miss_tokens * cache_miss_price
        + completion_tokens * output_price
    ) / 1_000_000
