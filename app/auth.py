"""Minimal stdlib-only JWT authentication helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any


SECRET_KEY = os.getenv(
    "AUTH_SECRET", "sih26104-dev-secret-change-me-before-deploy"
)
TOKEN_TTL_S = int(os.getenv("TOKEN_TTL_S", "3600"))


class AuthError(Exception):
    """Base class for token authentication failures."""


class TokenExpired(AuthError):
    """Raised when a token's expiration time has passed."""


class InvalidToken(AuthError):
    """Raised when a token is malformed, invalid, or not yet active."""


def _encode_segment(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _decode_segment(segment: str) -> bytes:
    if not segment or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for character in segment):
        raise InvalidToken("Malformed token")
    try:
        return base64.b64decode(
            segment + "=" * (-len(segment) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise InvalidToken("Malformed token") from exc


def create_token(
    subject: str,
    ttl_s: int = TOKEN_TTL_S,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create an HS256 JWT for ``subject``."""
    issued_at = int(time.time())
    claims: dict[str, Any] = dict(extra_claims or {})
    claims.update(
        {
            "sub": subject,
            "iat": issued_at,
            "nbf": issued_at,
            "exp": issued_at + int(ttl_s),
            "jti": uuid.uuid4().hex,
        }
    )

    header = _encode_segment({"alg": "HS256", "typ": "JWT"})
    payload = _encode_segment(claims)
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    signature_segment = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{header}.{payload}.{signature_segment}"


def decode_token(token: str) -> dict[str, Any]:
    """Verify and decode an HS256 JWT."""
    if not isinstance(token, str):
        raise InvalidToken("Malformed token")

    segments = token.split(".")
    if len(segments) != 3:
        raise InvalidToken("Malformed token")

    header_segment, payload_segment, signature_segment = segments
    try:
        header = json.loads(_decode_segment(header_segment))
        claims = json.loads(_decode_segment(payload_segment))
        signature = _decode_segment(signature_segment)
    except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise InvalidToken("Malformed token") from exc

    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise InvalidToken("Unsupported signing algorithm")
    if not isinstance(claims, dict):
        raise InvalidToken("Malformed claims")

    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.new(
        SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidToken("Invalid signature")

    now = int(time.time())
    try:
        expiration = claims["exp"]
        not_before = claims["nbf"]
        if not isinstance(expiration, (int, float)) or isinstance(expiration, bool):
            raise TypeError
        if not isinstance(not_before, (int, float)) or isinstance(not_before, bool):
            raise TypeError
    except (KeyError, TypeError) as exc:
        raise InvalidToken("Missing or invalid time claims") from exc

    if expiration <= now:
        raise TokenExpired("Token has expired")
    if not_before > now:
        raise InvalidToken("Token is not active")

    return claims


def subject_from_token(token: str) -> str | None:
    """Return a token subject, or ``None`` when no subject claim is present."""
    subject = decode_token(token).get("sub")
    return subject if isinstance(subject, str) else None


async def websocket_authenticate(websocket: Any) -> dict[str, Any] | None:
    """Authenticate a WebSocket using a query token or Bearer header."""
    token = websocket.query_params.get("token")
    if not token:
        authorization = websocket.headers.get("authorization", "")
        scheme, _, header_token = authorization.partition(" ")
        if scheme.lower() == "bearer" and header_token:
            token = header_token

    if not token:
        return None
    return decode_token(token)