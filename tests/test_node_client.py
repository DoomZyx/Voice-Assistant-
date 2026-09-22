import httpx
import pytest

from app.config import Settings
from app.node_client import NodeApiClient, NodeApiError


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/ping":
        return httpx.Response(200, json={"success": True, "message": "Backend actif"})
    if path == "/api/voice/context/restaurant-42":
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "restaurantInfo": {"nom": "Restaurant Test"},
                    "menu": {},
                },
            },
        )
    if path == "/api/orders/ai/available-slots":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(200, json={"availableSlots": ["12:00", "12:30"], "message": "ok"})
    if path == "/api/reservations/ai/available-slots":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(200, json={"remainingCoversMidi": 8, "remainingCoversSoir": 12})
    if path == "/api/reservations/ai/create":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(
            201,
            json={"success": True, "data": {"_id": "resa1", "nom": "Axel"}, "message": "Réservation créée"},
        )
    if path == "/api/orders/ai/create":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(
            201,
            json={"success": True, "data": {"_id": "ord1"}, "message": "Commande créée"},
        )
    if path == "/api/voice/usage":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(204)
    if path == "/api/voice/call-ended":
        assert request.headers.get("x-tenant-id") == "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
        return httpx.Response(200, json={"success": True, "persisted": False, "notified": True})
    return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def settings() -> Settings:
    return Settings(node_base_url="http://node.test", node_api_key="test-key")


@pytest.fixture
async def client(settings: Settings):
    transport = httpx.MockTransport(_handler)
    http = httpx.AsyncClient(transport=transport, base_url=settings.node_base_url)
    node = NodeApiClient(settings, client=http)
    yield node
    await node.aclose()


@pytest.mark.asyncio
async def test_ping(client: NodeApiClient):
    data = await client.ping()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_get_voice_context(client: NodeApiClient):
    data = await client.get_voice_context("restaurant-42")
    assert data["restaurantInfo"]["nom"] == "Restaurant Test"


@pytest.mark.asyncio
async def test_check_availability(client: NodeApiClient):
    data = await client.check_availability("2026-09-09", tenant_id="ff3e05bf-2da5-465c-a7a2-f635021f49a9")
    assert data["success"] is True
    assert data["fenetres"] == [{"debut": "12:00", "fin": "12:30"}]
    assert "slots" not in data
    assert data["remainingCoversMidi"] == 8
    assert data["remainingCoversSoir"] == 12


@pytest.mark.asyncio
async def test_create_reservation(client: NodeApiClient):
    data = await client.create_appointment(
        {
            "type": "Réservation de table",
            "name": "Axel",
            "telephone": "06 72 88 62 55",
            "date": "2026-09-09",
            "time": "12:20",
            "nombrePersonnes": 6,
        },
        tenant_id="ff3e05bf-2da5-465c-a7a2-f635021f49a9",
    )
    assert data["success"] is True
    assert data["created"] is True
    assert data["id"] == "resa1"
    assert data["type"] == "reservation"
    assert "appointment" not in data


@pytest.mark.asyncio
async def test_create_order(client: NodeApiClient):
    data = await client.create_appointment(
        {
            "type": "Commande à emporter",
            "name": "Axel",
            "telephone": "06 72 88 62 55",
            "date": "2026-09-09",
            "time": "19:00",
            "commandes": [{"nom": "Burger", "quantite": 1}],
        },
        tenant_id="ff3e05bf-2da5-465c-a7a2-f635021f49a9",
    )
    assert data["success"] is True
    assert data["created"] is True
    assert data["id"] == "ord1"
    assert data["type"] == "order"
    assert "appointment" not in data


@pytest.mark.asyncio
async def test_create_reservation_http_error(settings: Settings):
    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "Numéro de téléphone invalide ou manquant."})

    http = httpx.AsyncClient(transport=httpx.MockTransport(fail), base_url=settings.node_base_url)
    node = NodeApiClient(settings, client=http)
    with pytest.raises(NodeApiError) as exc:
        await node.create_reservation(
            {"name": "Axel", "telephone": "1", "date": "2026-09-09", "time": "12:00"}
        )
    assert exc.value.status_code == 400
    await node.aclose()


@pytest.mark.asyncio
async def test_network_failure_becomes_node_api_error(settings: Settings):
    """Une panne reseau/timeout httpx doit devenir un NodeApiError taggue
    network_error, pas une exception httpx brute qui casserait la session."""

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(fail), base_url=settings.node_base_url)
    node = NodeApiClient(settings, client=http)
    with pytest.raises(NodeApiError) as exc:
        await node.check_availability("2026-09-09", tenant_id="ff3e05bf-2da5-465c-a7a2-f635021f49a9")
    assert exc.value.payload == {"network_error": True}
    await node.aclose()


@pytest.mark.asyncio
async def test_report_call_ended(client: NodeApiClient):
    data = await client.report_call_ended(
        {
            "instanceId": "ff3e05bf-2da5-465c-a7a2-f635021f49a9",
            "streamSid": "MZ1",
            "callSid": "CA1",
            "turns": [{"role": "user", "content": "oui"}],
            "liveCreate": {"type": "reservation", "id": "resa1"},
        },
        tenant_id="ff3e05bf-2da5-465c-a7a2-f635021f49a9",
    )
    assert data["success"] is True
    assert data["persisted"] is False
