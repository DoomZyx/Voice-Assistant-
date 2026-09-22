import pytest

from app.node_client import NodeApiError
from app.tools.dispatcher import ToolDispatcher, map_create_error
from tests.fakes import FakeNode


@pytest.mark.asyncio
async def test_dispatch_availability():
    node = FakeNode()
    dispatcher = ToolDispatcher(node)
    result = await dispatcher.dispatch("check_availability", {"date": "2026-09-09"})
    assert result["success"] is True
    assert node.availability_calls == ["2026-09-09"]


@pytest.mark.asyncio
async def test_dispatch_unknown():
    dispatcher = ToolDispatcher(FakeNode())
    result = await dispatcher.dispatch("drop_database", {})
    assert result["success"] is False


@pytest.mark.asyncio
async def test_dispatch_create():
    node = FakeNode()
    dispatcher = ToolDispatcher(node)
    result = await dispatcher.dispatch(
        "create_appointment",
        {
            "type": "Commande à emporter",
            "name": "Axel",
            "telephone": "06 72 88 62 55",
            "date": "2026-09-09",
            "time": "19:00",
        },
    )
    assert result["success"] is True
    assert len(node.appointments) == 1
    assert result["id"] == "ord1"
    assert result["type"] == "order"


@pytest.mark.asyncio
async def test_dispatch_create_allows_reservation_when_closed_now():
    """Une resa pour plus tard ne doit pas etre bloquee par isOpenNow=false."""
    node = FakeNode()
    node.voice_snapshots["tenant-1"] = {
        "availability": False,
        "restaurantOuvert": False,
        "products": [],
    }
    dispatcher = ToolDispatcher(node)
    result = await dispatcher.dispatch(
        "create_appointment",
        {
            "type": "Réservation de table",
            "name": "Axel",
            "telephone": "06 72 88 62 55",
            "date": "2026-09-09",
            "time": "19:00",
            "nombrePersonnes": 2,
        },
        tenant_id="tenant-1",
    )
    assert result["success"] is True
    assert len(node.appointments) == 1


@pytest.mark.asyncio
async def test_dispatch_create_blocks_missing_product():
    node = FakeNode()
    node.voice_snapshots["tenant-1"] = {
        "availability": True,
        "products": [{"nom": "Burger Classic", "categorie": "burgers"}],
    }
    dispatcher = ToolDispatcher(node)
    result = await dispatcher.dispatch(
        "create_appointment",
        {
            "type": "Commande à emporter",
            "name": "Axel",
            "telephone": "06 72 88 62 55",
            "date": "2026-09-09",
            "time": "19:00",
            "commandes": [{"nom": "Plat Invente", "quantite": 1}],
        },
        tenant_id="tenant-1",
    )
    assert result["success"] is False
    assert result["error"] == "PRODUIT_INDISPONIBLE"
    assert node.appointments == []


def test_map_phone_error():
    mapped = map_create_error(
        NodeApiError("bad", 400, {"error": "Numéro de téléphone invalide ou manquant."})
    )
    assert mapped["error"] == "NUMERO_MANQUANT"


def test_map_network_error_is_conversational_not_a_crash():
    """Une panne reseau Fastify (NodeApiError taggee network_error) doit devenir
    un code conversationnel, pas remonter comme une exception httpx brute."""
    mapped = map_create_error(
        NodeApiError("Fastify injoignable (ConnectTimeout)", None, {"network_error": True})
    )
    assert mapped["success"] is False
    assert mapped["error"] == "SERVICE_INDISPONIBLE"
    assert "ConnectTimeout" not in mapped["message"]
