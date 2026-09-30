import json

import httpx
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.tools.dispatcher import ToolDispatcher, session_tool_arguments
from livekit_agent.tools.http import FastifyToolsClient


RESTAURANT_CONTEXT = {
    "success": True,
    "data": {
        "restaurantInfo": {
            "nom": "Le Comptoir",
            "adresse": "1 rue Test",
            "telephone": "0102030405",
            "horairesOuverture": {"lundi": {"ouvert": True}},
        },
        "menu": {
            "burgers": {
                "nom": "Burgers",
                "produits": [
                    {"nom": "Burger Classic", "prix": 12.5, "description": "Boeuf"},
                    {"nom": "Burger Veggie", "prix": 11.0},
                ],
            }
        },
        "availability": True,
        "timezone": "Europe/Paris",
        "amenities": [
            {"slug": "pmr", "label": "Accès PMR", "status": "available"},
            {"slug": "highchair", "label": "Chaises bébé", "status": "available", "quantity": 3},
        ],
    },
}

SNAPSHOT = {
    "success": True,
    "data": {
        "availability": True,
        "restaurantOuvert": True,
        "products": [
            {"nom": "Burger Classic", "categorie": "burgers"},
            {"nom": "Burger Veggie", "categorie": "burgers"},
        ],
    },
}

SEEN: list[tuple[str, str, dict]] = []


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    body: dict = {}
    if request.content:
        try:
            parsed = json.loads(request.content)
            if isinstance(parsed, dict):
                body = parsed
        except ValueError:
            body = {}
    SEEN.append((request.method, path, body))
    if path == "/api/voice/context/tenant-1":
        return httpx.Response(200, json=RESTAURANT_CONTEXT)
    if path == "/api/voice/context/tenant-1/snapshot":
        return httpx.Response(200, json=SNAPSHOT)
    if path.endswith("/api/orders/ai/create") and request.method == "POST":
        return httpx.Response(201, json={"success": True, "data": {"_id": "order-01"}})
    if path.endswith("/api/orders/order-01/status") and request.method == "PATCH":
        return httpx.Response(200, json={"success": True, "data": {"_id": "order-01", "statut": body.get("statut")}})
    if request.method in {"PUT", "DELETE"}:
        return httpx.Response(405, json={"error": "staff route disabled for voice tools"})
    return httpx.Response(404, json={"error": "not found"})


def _client() -> FastifyToolsClient:
    SEEN.clear()
    settings = LiveKitSettings(node_base_url="http://fastify.test", node_api_key="secret")
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url="http://fastify.test",
        headers={"x-api-key": "secret", "Content-Type": "application/json"},
    )
    return FastifyToolsClient(settings, client=http)


@pytest.mark.asyncio
async def test_get_restaurant_info_strips_menu():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    result = await dispatcher.dispatch("get_restaurant_info", {})
    assert result["success"] is True
    assert result["data"]["nom"] == "Le Comptoir"
    assert "menu" not in result["data"]
    assert result["data"]["amenities"][1]["quantity"] == 3
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_menu_search_and_get_product():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    menu = await dispatcher.dispatch("get_menu", {})
    assert menu["success"] is True
    assert menu["data"]["burgers"]["produits"][0]["nom"] == "Burger Classic"

    found = await dispatcher.dispatch("search_product", {"query": "classic"})
    assert found["success"] is True
    assert len(found["data"]) == 1
    assert found["data"][0]["nom"] == "Burger Classic"

    missing = await dispatcher.dispatch("get_product", {"nom": "Plat Invente"})
    assert missing["success"] is False
    product = await dispatcher.dispatch("get_product", {"nom": "Burger Veggie"})
    assert product["success"] is True
    assert product["data"]["prix"] == 11.0
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_availability_and_orders():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    available = await dispatcher.dispatch(
        "check_product_availability",
        {"nom": "Burger Classic"},
    )
    assert available["success"] is True
    assert available["data"]["available"] is True

    created = await dispatcher.dispatch(
        "create_order",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-20",
            "time": "19:00",
            "commandes": [{"nom": "Burger Classic", "quantite": 1}],
        },
    )
    assert created["success"] is True
    assert created["id"] == "order-01"
    create_calls = [item for item in SEEN if item[0] == "POST" and item[1].endswith("/api/orders/ai/create")]
    assert len(create_calls) == 1
    assert create_calls[0][2]["type"] == "Commande à emporter"
    assert create_calls[0][2]["commandes"][0]["nom"] == "Burger Classic"

    updated = await dispatcher.dispatch("update_order", {"id": "order-01", "statut": "confirme"})
    assert updated["success"] is True
    cancelled = await dispatcher.dispatch("cancel_order", {"id": "order-01"})
    assert cancelled["success"] is True
    status_calls = [item for item in SEEN if item[0] == "PATCH" and item[1].endswith("/status")]
    assert status_calls[0][2] == {"statut": "confirme"}
    assert status_calls[1][2] == {"statut": "annule"}
    assert not any(method == "DELETE" for method, _path, _body in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_create_order_rejects_unknown_product():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    result = await dispatcher.dispatch(
        "create_order",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-20",
            "time": "19:00",
            "commandes": [{"nom": "Plat Invente", "quantite": 1}],
        },
    )
    assert result["success"] is False
    assert result["error"] == "PRODUIT_INDISPONIBLE"
    assert not any(path.endswith("/api/orders/ai/create") for _method, path, _body in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    unknown = await dispatcher.dispatch("drop_database", {})
    assert unknown["success"] is False
    await dispatcher.aclose()


def test_session_tool_arguments_strip_nested_tenant_selectors():
    cleaned = session_tool_arguments(
        {
            "name": "Ada",
            "restaurant_id": "tenant-b",
            "tenantId": "tenant-b",
            "instanceId": "tenant-b",
            "schema": "tenant_b",
            "commandes": [
                {"nom": "Burger", "restaurant_id": "tenant-b", "instance_id": "tenant-b"},
            ],
        }
    )
    assert cleaned == {"name": "Ada", "commandes": [{"nom": "Burger"}]}
    assert "restaurant_id" not in cleaned
    assert "tenantId" not in cleaned


@pytest.mark.asyncio
async def test_llm_cannot_switch_session_tenant():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-1")
    result = await dispatcher.dispatch(
        "get_restaurant_info",
        {"restaurant_id": "tenant-2", "tenant_id": "tenant-2", "instanceId": "tenant-2"},
    )
    assert result["success"] is True
    assert result["data"]["nom"] == "Le Comptoir"
    assert all("/tenant-2" not in path for _method, path, _body in SEEN)
    assert any(path.endswith("/api/voice/context/tenant-1") for _method, path, _body in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_did_tenant_a_cannot_read_or_order_as_b():
    dispatcher = ToolDispatcher(_client(), tenant_id="tenant-2")
    info = await dispatcher.dispatch("get_restaurant_info", {})
    assert info["success"] is False
    created = await dispatcher.dispatch(
        "create_order",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-20",
            "time": "19:00",
            "restaurant_id": "tenant-1",
            "commandes": [{"nom": "Burger Classic", "quantite": 1}],
        },
    )
    assert created["success"] is False
    assert not any(path.endswith("/api/orders/ai/create") for _method, path, _body in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_unresolved_tenant_never_calls_fastify():
    dispatcher = ToolDispatcher(_client(), tenant_id="")
    result = await dispatcher.dispatch("get_restaurant_info", {"restaurant_id": "tenant-1"})
    assert result["success"] is False
    assert result["error"] == "TENANT_UNRESOLVED"
    assert SEEN == []
    await dispatcher.aclose()
