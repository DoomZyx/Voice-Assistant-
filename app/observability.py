from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger("mysmartfood-voice.trace")

# Les evenements sont visibles dans les journaux d'exploitation. Ils ne sont
# donc jamais un canal pour le contenu d'un appel ni pour des credentials.
_SENSITIVE_FIELD_PARTS = (
    "audio", "authorization", "caller", "content", "key", "message",
    "phone", "prompt", "secret", "telephone", "token", "transcript",
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){8,}\d(?!\d)")
_BEARER_RE = re.compile(r"\bbearer\s+[^\s]+", re.IGNORECASE)


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Conserve les metadonnees operationnelles, jamais les donnees d'appel."""
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        normalized_key = str(key).lower()
        if any(part in normalized_key for part in _SENSITIVE_FIELD_PARTS):
            continue
        if isinstance(value, str):
            sanitized = _BEARER_RE.sub("[redacted-credential]", value)
            sanitized = _PHONE_RE.sub("[redacted-phone]", sanitized)
            safe[key] = sanitized[:200]
        else:
            safe[key] = value
    return safe


def emit_event(
    step: str,
    *,
    call_id: str | None = None,
    session_id: str | None = None,
    latency_ms: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    **fields: Any,
) -> None:
    parts = [f"step={step}"]
    if call_id:
        parts.append(f"call_id={call_id}")
    if session_id:
        parts.append(f"session_id={session_id}")
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms}")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    for key, value in sanitize_log_fields(fields).items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))
