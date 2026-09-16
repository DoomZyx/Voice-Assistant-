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


def test_map_phone_error():
    mapped = map_create_error(
        NodeApiError("bad", 400, {"error": "Numéro de téléphone invalide ou manquant."})
    )
    assert mapped["error"] == "NUMERO_MANQUANT"
