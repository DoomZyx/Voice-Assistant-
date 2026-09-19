from __future__ import annotations

import unicodedata
from typing import Any

from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


def flatten_products(menu: Any) -> list[dict[str, Any]]:
    if not isinstance(menu, dict):
        return []
    products: list[dict[str, Any]] = []
    for slug, category in menu.items():
        if not isinstance(category, dict):
            continue
        category_name = category.get("nom") or slug
        for item in category.get("produits") or []:
            if not isinstance(item, dict) or not str(item.get("nom") or "").strip():
                continue
            products.append({**item, "categorie": category_name})
    return products


async def _load_menu(client: FastifyToolsClient, tenant_id: str) -> dict[str, Any]:
    context = await client.get_voice_context(tenant_id)
    menu = context.get("menu")
    return menu if isinstance(menu, dict) else {}


async def get_menu(client: FastifyToolsClient, tenant_id: str) -> dict[str, Any]:
    try:
        menu = await _load_menu(client, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "data": menu}


async def search_product(client: FastifyToolsClient, tenant_id: str, query: str) -> dict[str, Any]:
    needle = _norm(query)
    if not needle:
        return {"success": False, "error": "QUERY_REQUIRED"}
    try:
        menu = await _load_menu(client, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
    matches = [item for item in flatten_products(menu) if needle in _norm(item.get("nom"))]
    return {"success": True, "data": matches}


async def get_product(client: FastifyToolsClient, tenant_id: str, name: str) -> dict[str, Any]:
    needle = _norm(name)
    if not needle:
        return {"success": False, "error": "NAME_REQUIRED"}
    try:
        menu = await _load_menu(client, tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
    for item in flatten_products(menu):
        if _norm(item.get("nom")) == needle:
            return {"success": True, "data": item}
    return {"success": False, "error": "PRODUCT_NOT_FOUND"}


async def check_product_availability(
    client: FastifyToolsClient,
    tenant_id: str,
    name: str,
) -> dict[str, Any]:
    needle = _norm(name)
    if not needle:
        return {"success": False, "error": "NAME_REQUIRED"}
    try:
        snapshot = await client.get_voice_snapshot(tenant_id)
    except FastifyApiError as exc:
        return {"success": False, "error": str(exc)}
    products = snapshot.get("products") if isinstance(snapshot.get("products"), list) else []
    available = any(isinstance(item, dict) and _norm(item.get("nom")) == needle for item in products)
    return {
        "success": True,
        "data": {
            "available": available,
            "restaurantOuvert": bool(snapshot.get("restaurantOuvert") or snapshot.get("availability")),
        },
    }
