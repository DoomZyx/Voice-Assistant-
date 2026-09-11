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
        return httpx.Response(200, json={"availableSlots": ["12:00", "12:30"], "message": "ok"})
    if path == "/api/reservations/ai/available-slots":
        return httpx.Response(200, json={"remainingCoversMidi": 8, "remainingCoversSoir": 12})
    if path == "/api/reservations/ai/create":
        return httpx.Response(
            201,
            json={"success": True, "data": {"_id": "resa1", "nom": "Axel"}, "message": "Réservation créée"},
        )
    if path == "/api/orders/ai/create":
        return httpx.Response(
            201,
            json={"success": True, "data": {"_id": "ord1"}, "message": "Commande créée"},
        )
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
    data = await client.check_availability("2026-09-09")
    assert data["success"] is True
    assert data["slots"] == ["12:00", "12:30"]
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
        }
    )
    assert data["success"] is True
    assert data["appointment"]["_id"] == "resa1"


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
        }
    )
    assert data["success"] is True
    assert data["appointment"]["_id"] == "ord1"


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
