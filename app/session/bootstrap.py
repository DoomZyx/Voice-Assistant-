"""Contrat d'ouverture d'appel partage par CallSession et Pipecat."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm.tools import LLM_TOOLS
from app.prompts import (
    build_agent_instructions,
    build_greeting,
    has_usable_restaurant_context,
)
from app.stream_token import stream_token_failure_reason

VOICE_CONTEXT_TIMEOUT_SECONDS = 3.0
MASKED_CALLER_VALUES = {"anonymous", "blocked", "private", "restricted", "unknown", "withheld"}


@dataclass(frozen=True)
class CallPlan:
    stream_sid: str
    call_sid: str
    instance_id: str | None
    caller_number: str | None
    rejected: bool
    reject_reason: str | None
    context_ready: bool
    context_unavailable: bool
    last_error: str | None
    system_prompt: str
    greeting: str
    seed_messages: list[dict[str, Any]]
    tools_enabled: bool

    @property
    def llm_tools(self) -> list[dict[str, Any]]:
        return list(LLM_TOOLS) if self.tools_enabled else []


def apply_forced_caller_number(
    arguments: dict[str, Any],
    caller_number: str | None,
) -> dict[str, Any]:
    updated = dict(arguments)
    if caller_number:
        updated["telephone"] = caller_number
    return updated


def sanitize_identifier(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "", text)
    return sanitized[:128] or None


def normalize_french_caller_number(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in MASKED_CALLER_VALUES:
        return None
    compact = re.sub(r"[\s().-]", "", raw)
    if compact.startswith("0033"):
        compact = f"+33{compact[4:]}"
    if not re.fullmatch(r"\+33[1-9]\d{8}", compact):
        return None
    return f"0{compact[3:]}"


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def build_call_plan(
    start_event: dict[str, Any] | None,
    settings: Settings,
    node: Any,
    telemetry: Any | None = None,
) -> CallPlan:
    start = (start_event or {}).get("start") or {}
    custom = start.get("customParameters") or {}
    if not isinstance(custom, dict):
        custom = {}
    stream_sid = str(start.get("streamSid") or "")
    call_sid = str(start.get("callSid") or "")
    caller_number = normalize_french_caller_number(custom.get("callerNumber"))
    requested_id = sanitize_identifier(custom.get("instanceId"))
    generic_prompt = build_agent_instructions()

    if requested_id:
        token = (
            custom.get("streamToken")
            or custom.get("token")
            or custom.get("StreamToken")
            or ""
        )
        secret = str(settings.twilio_auth_token or "").strip()
        reason = stream_token_failure_reason(requested_id, token, secret)
        if reason:
            return CallPlan(
                stream_sid=stream_sid,
                call_sid=call_sid,
                instance_id=None,
                caller_number=caller_number,
                rejected=True,
                reject_reason=reason,
                context_ready=False,
                context_unavailable=False,
                last_error="stream_token_rejected",
                system_prompt=generic_prompt,
                greeting="",
                seed_messages=[{"role": "system", "content": generic_prompt}],
                tools_enabled=False,
            )

    restaurant_context = None
    context_unavailable = False
    last_error = None
    if requested_id:
        started = time.perf_counter()
        try:
            restaurant_context = await asyncio.wait_for(
                node.get_voice_context(requested_id),
                timeout=VOICE_CONTEXT_TIMEOUT_SECONDS,
            )
        except Exception:
            context_unavailable = True
            last_error = "voice_context_unavailable"
            if telemetry:
                telemetry.record_error("voice_context")
        finally:
            if telemetry:
                telemetry.record_latency("voice_context", _ms(started))

    if context_unavailable or (
        requested_id and not has_usable_restaurant_context(restaurant_context)
    ):
        context_unavailable = True
        context_ready = False
        if last_error is None:
            last_error = "voice_context_unavailable"
    else:
        context_ready = True

    system_prompt = build_agent_instructions(
        caller_number,
        restaurant_context,
        context_unavailable=context_unavailable and bool(requested_id),
    )
    greeting = build_greeting(
        restaurant_context,
        settings.greeting_text,
        context_unavailable=context_unavailable and bool(requested_id),
    )
    return CallPlan(
        stream_sid=stream_sid,
        call_sid=call_sid,
        instance_id=requested_id,
        caller_number=caller_number,
        rejected=False,
        reject_reason=None,
        context_ready=context_ready,
        context_unavailable=context_unavailable,
        last_error=last_error,
        system_prompt=system_prompt,
        greeting=greeting,
        seed_messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": greeting},
        ],
        tools_enabled=context_ready,
    )
