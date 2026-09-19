from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from livekit_agent.config import LiveKitSettings


@dataclass(frozen=True)
class CallIdentity:
    call_id: str
    session_id: str
    restaurant_id: str
    caller_number: str | None = None
    trunk_number: str | None = None


def normalize_phone(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit() or ch == "+")


def parse_did_map(raw: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {normalize_phone(str(key)): str(value) for key, value in payload.items() if str(value).strip()}


def _attr(attributes: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = attributes.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_call_identity(
    settings: LiveKitSettings,
    *,
    room_name: str,
    sip_attributes: dict[str, Any] | None = None,
) -> CallIdentity:
    attributes = sip_attributes or {}
    call_id = _attr(attributes, "sip.callID", "sip.callId", "sip.twilio.callSid") or room_name
    caller = normalize_phone(_attr(attributes, "sip.phoneNumber", "sip.from"))
    trunk = normalize_phone(_attr(attributes, "sip.trunkPhoneNumber", "sip.to"))
    did_map = parse_did_map(settings.sip_did_tenant_map)
    restaurant_id = did_map.get(trunk) or did_map.get(caller) or settings.restaurant_id or ""
    return CallIdentity(
        call_id=call_id,
        session_id=room_name,
        restaurant_id=restaurant_id,
        caller_number=caller or None,
        trunk_number=trunk or None,
    )


def first_sip_attributes(room: Any) -> dict[str, Any]:
    participants = getattr(room, "remote_participants", None)
    if not participants:
        return {}
    values = participants.values() if hasattr(participants, "values") else participants
    for participant in values:
        attributes = getattr(participant, "attributes", None) or {}
        if attributes:
            return dict(attributes)
    return {}
