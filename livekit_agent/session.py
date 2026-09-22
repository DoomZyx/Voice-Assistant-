from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from livekit_agent.config import LiveKitSettings


JOB_IDENTITY_KEY = "mysmartfood_call_identity"


@dataclass(frozen=True)
class CallIdentity:
    call_id: str
    session_id: str
    restaurant_id: str
    caller_number: str | None = None
    trunk_number: str | None = None

    @property
    def tenant_id(self) -> str:
        return self.restaurant_id

    @property
    def instance_id(self) -> str:
        return self.restaurant_id


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
    attributes = {
        str(key): value
        for key, value in dict(sip_attributes or {}).items()
        if str(key).startswith("sip.")
    }
    call_id = _attr(attributes, "sip.callID", "sip.callId", "sip.twilio.callSid") or room_name
    caller = normalize_phone(_attr(attributes, "sip.phoneNumber", "sip.from"))
    trunk = normalize_phone(_attr(attributes, "sip.trunkPhoneNumber", "sip.to"))
    # SIP : le DID appelé (trunk) identifie l'instance. Jamais le numéro appelant.
    # restaurant_id vient ensuite de Fastify / twilio_bundles, pas d'une map locale.
    if trunk:
        restaurant_id = ""
    else:
        restaurant_id = settings.restaurant_id or ""
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
        sip_only = {
            str(key): value
            for key, value in dict(attributes).items()
            if str(key).startswith("sip.")
        }
        if sip_only:
            return sip_only
    return {}


def read_job_identity(ctx: Any) -> CallIdentity | None:
    userdata = getattr(ctx, "userdata", None)
    if isinstance(userdata, dict):
        stored = userdata.get(JOB_IDENTITY_KEY)
        if isinstance(stored, CallIdentity):
            return stored
    stored = getattr(ctx, JOB_IDENTITY_KEY, None)
    return stored if isinstance(stored, CallIdentity) else None


def bind_job_identity(ctx: Any, identity: CallIdentity) -> CallIdentity:
    existing = read_job_identity(ctx)
    if existing is not None:
        return existing
    userdata = getattr(ctx, "userdata", None)
    if not isinstance(userdata, dict):
        userdata = {}
        ctx.userdata = userdata
    userdata[JOB_IDENTITY_KEY] = identity
    setattr(ctx, JOB_IDENTITY_KEY, identity)
    ctx.log_context_fields = {
        "room": identity.session_id,
        "call_id": identity.call_id,
        "session_id": identity.session_id,
        "restaurant_id": identity.restaurant_id,
        "tenant_id": identity.tenant_id,
        "instance_id": identity.instance_id,
    }
    return identity
