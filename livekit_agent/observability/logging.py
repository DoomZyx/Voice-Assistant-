from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mysmartfood-voice.livekit")


def emit_event(
    step: str,
    *,
    call_id: str | None = None,
    session_id: str | None = None,
    restaurant_id: str | None = None,
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
    if restaurant_id:
        parts.append(f"restaurant_id={restaurant_id}")
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms}")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    line = " ".join(parts)
    if step.endswith("mismatch") or step.endswith("error"):
        logger.error(line)
        return
    logger.info(line)
