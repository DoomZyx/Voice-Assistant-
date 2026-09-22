from __future__ import annotations

from dataclasses import replace

from livekit_agent.config import LiveKitSettings
from livekit_agent.observability.logging import emit_event
from livekit_agent.session import CallIdentity
from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient


async def resolve_session_identity(
    settings: LiveKitSettings,
    identity: CallIdentity,
    client: FastifyToolsClient | None = None,
) -> CallIdentity:
    if not identity.trunk_number:
        return identity
    owns = client is None
    http = client or FastifyToolsClient(settings)
    try:
        payload = await http.get_inbound_tenant(identity.trunk_number)
        restaurant_id = str(payload.get("restaurantId") or payload.get("restaurant_id") or "").strip()
        if not restaurant_id:
            emit_event("tenant.unresolved", reason="empty")
            return identity
        emit_event("tenant.resolved", restaurant_id=restaurant_id)
        return replace(identity, restaurant_id=restaurant_id)
    except FastifyApiError as exc:
        emit_event(
            "tenant.unresolved",
            reason=str(exc),
            status=exc.status_code,
        )
        return identity
    finally:
        if owns:
            await http.aclose()
