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
    data = dict(info)
    amenities = context.get("amenities")
    if isinstance(amenities, list):
        data["amenities"] = [
            {key: item[key] for key in ("slug", "label", "status", "quantity") if key in item}
            for item in amenities
            if isinstance(item, dict)
        ]
    return {"success": True, "data": data}
