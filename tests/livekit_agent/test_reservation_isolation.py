import json

import httpx
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.session import CallIdentity
from livekit_agent.tools.dispatcher import LAST_RESERVATION_ID_KEY, ToolDispatcher
from livekit_agent.tools.http import FastifyToolsClient

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
RESA_A = "11111111-1111-1111-1111-111111111111"
RESA_B = "22222222-2222-2222-2222-222222222222"

SEEN: list[tuple[str, str, dict, dict]] = []


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    query = dict(request.url.params)
    body: dict = {}
    if request.content:
        try:
            parsed = json.loads(request.content)
            if isinstance(parsed, dict):
                body = parsed
        except ValueError:
            body = {}
    headers = {key.lower(): value for key, value in request.headers.items()}
    SEEN.append((request.method, path, body, {"headers": headers, "query": query}))
    tenant = headers.get("x-tenant-id", "")

    if path == f"/api/voice/context/{TENANT_A}":
        return httpx.Response(
            200,
            json={"success": True, "data": {"restaurantInfo": {"nom": "Resto A"}}},
        )
    if path == f"/api/voice/context/{TENANT_B}":
        return httpx.Response(
            200,
            json={"success": True, "data": {"restaurantInfo": {"nom": "Resto B"}}},
        )
    if path.startswith("/api/voice/context/"):
        return httpx.Response(403, json={"error": "tenant mismatch"})

    if path == "/api/reservations/availability":
        if tenant != TENANT_A:
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(
            200,
            json={"success": True, "available": query.get("heure") == "20:00", "remainingCoversSoir": 8},
        )
    if path == "/api/reservations/ai/available-slots":
        if tenant != TENANT_A:
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(
            200,
            json={
                "success": True,
                "availableSlots": ["19:30", "20:00", "20:30"],
                "remainingCoversSoir": 8,
                "message": "ok",
            },
        )
    if path == "/api/reservations/ai/create" and request.method == "POST":
        if tenant != TENANT_A:
            return httpx.Response(403, json={"error": "forbidden"})
        if "instanceId" in body or "restaurant_id" in body or "tenant_id" in body:
            return httpx.Response(400, json={"error": "tenant leaked in body"})
        return httpx.Response(
            201,
            json={"success": True, "data": {"id": RESA_A, "nom": body.get("name")}},
        )
    if path == f"/api/reservations/{RESA_A}" and request.method == "PUT":
        if tenant != TENANT_A:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"success": True, "data": {"id": RESA_A}})
    if path == f"/api/reservations/{RESA_A}/status" and request.method == "PATCH":
        if tenant != TENANT_A:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(
            200,
            json={"success": True, "data": {"id": RESA_A, "statut": body.get("statut")}},
        )
    if path == "/api/reservations" and request.method == "GET":
        if tenant == TENANT_B:
            return httpx.Response(403, json={"error": "forbidden", "data": []})
        if tenant == TENANT_A:
            return httpx.Response(
                200,
                json={"success": True, "data": [{"id": RESA_A, "nom": "Ada"}]},
            )
        return httpx.Response(404, json={"error": "not found"})
    if path.startswith("/api/reservations/") and request.method in {"GET", "PUT", "PATCH"}:
        return httpx.Response(404, json={"error": "not found"})
    return httpx.Response(404, json={"error": "not found"})


def _client() -> FastifyToolsClient:
    SEEN.clear()
    settings = LiveKitSettings(
        _env_file=None,
        node_base_url="http://fastify.test",
        node_api_key="secret",
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url="http://fastify.test",
        headers={"x-api-key": "secret", "Content-Type": "application/json"},
    )
    return FastifyToolsClient(settings, client=http)


def _dispatcher(tenant_id: str, session_store: dict | None = None) -> ToolDispatcher:
    return ToolDispatcher(
        _client(),
        tenant_id,
        session_store=session_store if session_store is not None else {},
        profile="reservations",
    )


def _tenant_headers() -> list[str]:
    return [item[3]["headers"].get("x-tenant-id", "") for item in SEEN]


@pytest.mark.asyncio
async def test_did_a_only_calls_fastify_as_a():
    dispatcher = _dispatcher(TENANT_A)
    info = await dispatcher.dispatch(
        "get_restaurant_info",
        {"restaurant_id": TENANT_B, "instanceId": TENANT_B, "tenant_id": TENANT_B},
    )
    slots = await dispatcher.dispatch(
        "check_table_availability",
        {"date": "2026-09-21", "heure": "20:00", "restaurant_id": TENANT_B},
    )
    assert info["success"] is True
    assert info["data"]["nom"] == "Resto A"
    assert slots["success"] is True
    assert slots["data"]["available"] is True
    assert _tenant_headers() == [TENANT_A, TENANT_A]
    assert all(TENANT_B not in path for _method, path, _body, _meta in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_tool_args_cannot_switch_reservation_tenant():
    dispatcher = _dispatcher(TENANT_A)
    created = await dispatcher.dispatch(
        "create_reservation",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-21",
            "time": "20:00",
            "nombrePersonnes": 4,
            "restaurant_id": TENANT_B,
            "instanceId": TENANT_B,
            "tenant_id": TENANT_B,
        },
    )
    assert created["success"] is True
    assert created["id"] == RESA_A
    create_calls = [item for item in SEEN if item[1].endswith("/api/reservations/ai/create")]
    assert len(create_calls) == 1
    assert create_calls[0][2].get("telephone") == "0600000000"
    assert "instanceId" not in create_calls[0][2]
    assert "restaurant_id" not in create_calls[0][2]
    assert create_calls[0][3]["headers"].get("x-tenant-id") == TENANT_A
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_create_a_then_list_b_is_empty():
    client_a = _client()
    created = await client_a.create_reservation(
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-21",
            "time": "20:00",
            "nombrePersonnes": 4,
        },
        TENANT_A,
    )
    assert created["id"] == RESA_A
    await client_a.aclose()

    client_b = _client()
    listed = await client_b.list_reservations(TENANT_B, date="2026-09-21")
    assert listed == []
    assert any(item[3]["headers"].get("x-tenant-id") == TENANT_B for item in SEEN)
    assert not any(
        item[1] == "/api/reservations" and item[3]["headers"].get("x-tenant-id") == TENANT_A
        for item in SEEN
    )
    await client_b.aclose()


@pytest.mark.asyncio
async def test_last_reservation_id_refused_across_sessions():
    store_a = {}
    dispatcher_a = _dispatcher(TENANT_A, session_store=store_a)
    created = await dispatcher_a.dispatch(
        "create_reservation",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-21",
            "time": "20:00",
            "nombrePersonnes": 4,
        },
    )
    assert created["id"] == RESA_A
    assert store_a[LAST_RESERVATION_ID_KEY] == RESA_A
    assert not hasattr(CallIdentity(call_id="c", session_id="s", restaurant_id=TENANT_A), LAST_RESERVATION_ID_KEY)

    store_b = {}
    dispatcher_b = _dispatcher(TENANT_B, session_store=store_b)
    replayed = await dispatcher_b.dispatch(
        "modify_reservation",
        {"id": RESA_A, "heure": "21:00", "restaurant_id": TENANT_A},
    )
    assert replayed["success"] is False
    assert replayed["error"] == "RESERVATION_ID_DENIED"
    assert not any(path.endswith(f"/api/reservations/{RESA_A}") for _m, path, _b, _meta in SEEN)

    foreign = await dispatcher_a.dispatch("modify_reservation", {"id": RESA_B, "heure": "21:00"})
    assert foreign["success"] is False
    assert foreign["error"] == "RESERVATION_ID_DENIED"

    updated = await dispatcher_a.dispatch("modify_reservation", {"id": RESA_A, "heure": "21:00"})
    assert updated["success"] is True
    cancelled = await dispatcher_a.dispatch("cancel_reservation", {"id": RESA_A})
    assert cancelled["success"] is True
    await dispatcher_a.aclose()
    await dispatcher_b.aclose()


@pytest.mark.asyncio
async def test_list_available_slots_returns_one_alternative():
    dispatcher = _dispatcher(TENANT_A)
    result = await dispatcher.dispatch(
        "list_available_slots",
        {"date": "2026-09-21", "heure": "20:00"},
    )
    assert result["success"] is True
    assert result["data"]["alternative"] == "19:30"
    assert "availableSlots" not in result["data"]
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_reservation_profile_hides_order_tools():
    dispatcher = _dispatcher(TENANT_A)
    result = await dispatcher.dispatch(
        "create_order",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-21",
            "time": "19:00",
            "commandes": [{"nom": "Burger", "quantite": 1}],
        },
    )
    assert result["success"] is False
    assert result["error"] == "TOOL_DISABLED"
    assert not any(path.endswith("/api/orders/ai/create") for _m, path, _b, _meta in SEEN)
    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_backend_failure_does_not_invent_confirmation():
    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    settings = LiveKitSettings(_env_file=None, node_base_url="http://fastify.test")
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(timeout_handler),
        base_url="http://fastify.test",
    )
    dispatcher = ToolDispatcher(
        FastifyToolsClient(settings, client=http),
        TENANT_A,
        session_store={},
        profile="reservations",
    )
    result = await dispatcher.dispatch(
        "create_reservation",
        {
            "name": "Ada",
            "telephone": "0600000000",
            "date": "2026-09-21",
            "time": "20:00",
        },
    )
    assert result["success"] is False
    assert result["error"] == "BACKEND_UNAVAILABLE"
    assert "enregistrer" in result["message"]
    await dispatcher.aclose()
