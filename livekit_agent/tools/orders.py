from __future__ import annotations

from typing import Any

from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient
from livekit_agent.tools.menu import _norm


async def _reject_unknown_products(
    client: FastifyToolsClient,
    tenant_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    commandes = arguments.get("commandes") or []
    if not isinstance(commandes, list) or not commandes:
        return None
    try:
        snapshot = await client.get_voice_snapshot(tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": "SNAPSHOT_INDISPONIBLE", "detail": str(exc)}
    products = snapshot.get("products") if isinstance(snapshot.get("products"), list) else []
    available = {
        _norm(item.get("nom"))
        for item in products
        if isinstance(item, dict) and item.get("nom")
    }
    missing = [
        str(line.get("nom") or "").strip()
        for line in commandes
        if isinstance(line, dict)
        and str(line.get("nom") or "").strip()
        and _norm(line.get("nom")) not in available
    ]
    if missing:
        return {"success": False, "error": "PRODUIT_INDISPONIBLE", "missing": missing}
    return None


async def create_order(client: FastifyToolsClient, tenant_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    gate = await _reject_unknown_products(client, tenant_id, arguments)
    if gate is not None:
        return gate
    try:
        return await client.create_order(arguments, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}


async def update_order(client: FastifyToolsClient, tenant_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    order_id = str(arguments.get("id") or arguments.get("order_id") or "").strip()
    if not order_id:
        return {"success": False, "error": "ORDER_ID_REQUIRED"}
    payload = {key: value for key, value in arguments.items() if key not in {"id", "order_id"}}
    try:
        return await client.update_order(order_id, payload, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}


async def cancel_order(client: FastifyToolsClient, tenant_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    order_id = str(arguments.get("id") or arguments.get("order_id") or "").strip()
    if not order_id:
        return {"success": False, "error": "ORDER_ID_REQUIRED"}
    try:
        return await client.cancel_order(order_id, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
