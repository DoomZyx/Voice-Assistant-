"""Jeton media-stream, même contrat que backend/utils/streamToken.js."""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import unquote


def create_stream_token(tenant_id: str, secret: str, now: int | None = None, ttl_seconds: int = 300) -> str:
    expires_at = int(now if now is not None else time.time()) + ttl_seconds
    payload = f"{tenant_id}.{expires_at}"
    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{signature}"


def stream_token_failure_reason(tenant_id: str, token: object, secret: str) -> str | None:
    provided = _decode_token(token)
    if not provided:
        return "token absent"
    parts = provided.split(".")
    if len(parts) != 3:
        return f"token mal forme ({len(parts)} parties)"
    token_tenant_id, expires_at, provided_signature = parts
    if token_tenant_id != str(tenant_id or ""):
        return "tenant mismatch"
    try:
        if int(expires_at) < int(time.time()):
            return "token expire"
    except ValueError:
        return "token mal forme (3 parties)"
    cleaned_secret = str(secret or "").strip()
    if not cleaned_secret:
        return "secret stream manquant"
    expected = hmac.new(
        cleaned_secret.encode("utf-8"),
        f"{token_tenant_id}.{expires_at}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, provided_signature):
        return "signature invalide"
    return None


def _decode_token(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return unquote(raw)
    except Exception:
        return raw
