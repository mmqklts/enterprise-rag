import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_JWT_SECRET_KEY = "dev-only-change-me"
INSECURE_JWT_SECRET_KEYS = {
    DEFAULT_JWT_SECRET_KEY,
    "replace_with_a_long_random_secret",
}
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", DEFAULT_JWT_SECRET_KEY)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LENGTH = 32


class AuthError(ValueError):
    pass


def validate_jwt_secret_config(secret: str | None = None) -> None:
    app_environment = os.getenv("APP_ENV", "development").strip().lower()
    if app_environment != "production":
        return

    configured_secret = secret if secret is not None else JWT_SECRET_KEY
    if (
        configured_secret in INSECURE_JWT_SECRET_KEYS
        or len(configured_secret) < 32
    ):
        raise RuntimeError(
            "生产环境必须配置至少 32 字符且非默认值的 JWT_SECRET_KEY"
        )


validate_jwt_secret_config()


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_KEY_LENGTH,
    )
    return (
        f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}$"
        f"{_base64url_encode(salt)}${_base64url_encode(digest)}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_value, digest_value = encoded.split("$")
        if algorithm != "scrypt":
            return False

        salt = _base64url_decode(salt_value)
        expected = _base64url_decode(digest_value)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False

    return hmac.compare_digest(actual, expected)


def create_access_token(
    user_id: int,
    username: str,
    expires_minutes: int | None = None,
) -> str:
    now = int(time.time())
    lifetime = (
        expires_minutes
        if expires_minutes is not None
        else ACCESS_TOKEN_EXPIRE_MINUTES
    )
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + lifetime * 60,
    }
    header_part = _base64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    payload_part = _base64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{header_part}.{payload_part}.{_base64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected_signature = hmac.new(
            JWT_SECRET_KEY.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_signature = _base64url_decode(signature_part)

        if not hmac.compare_digest(actual_signature, expected_signature):
            raise AuthError("Token 签名无效")

        header = json.loads(
            _base64url_decode(header_part).decode("utf-8")
        )
        payload = json.loads(
            _base64url_decode(payload_part).decode("utf-8")
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AuthError("Token 格式无效") from exc

    if header.get("alg") != JWT_ALGORITHM:
        raise AuthError("Token 算法无效")
    if int(payload.get("exp", 0)) <= int(time.time()):
        raise AuthError("Token 已过期")
    if not str(payload.get("sub", "")).isdigit():
        raise AuthError("Token 用户信息无效")

    return payload
