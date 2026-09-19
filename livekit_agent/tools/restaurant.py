from __future__ import annotations

from typing import Any

from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient


async def get_restaurant_info(client: FastifyToolsClient, tenant_id: str) -> dict[str, Any]:
    try:
        context = await client.get_voice_context(tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
    info = context.get("restaurantInfo")
    if not isinstance(info, dict):
        return {"success": False, "error": "RESTAURANT_INFO_UNAVAILABLE"}
    return {"success": True, "data": dict(info)}
