from typing import Any

import pytest


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(arguments or {})
        self.calls.append((name, payload))
        return {"success": True, "tool": name, "arguments": payload}


@pytest.mark.asyncio
async def test_livekit_tools_dispatch_names():
    pytest.importorskip("livekit.agents")
    from livekit_agent.tools.livekit_tools import build_livekit_tools

    dispatcher = FakeDispatcher()
    tools = build_livekit_tools(dispatcher)  # type: ignore[arg-type]
    names = [str(getattr(tool, "__name__", getattr(tool, "name", tool))) for tool in tools]
    assert any("get_restaurant_info" in name for name in names)
    assert any("create_order" in name for name in names)
    assert any("cancel_order" in name for name in names)
    info = next(tool for tool in tools if "get_restaurant_info" in str(getattr(tool, "__name__", tool)))
    result = await info(None)
    assert result["success"] is True
    assert dispatcher.calls == [("get_restaurant_info", {})]


@pytest.mark.asyncio
async def test_reservation_tool_type_hints_resolve_run_context():
    pytest.importorskip("livekit.agents")
    from typing import get_type_hints

    from livekit.agents import RunContext

    from livekit_agent.tools.livekit_tools import build_livekit_tools

    tools = build_livekit_tools(FakeDispatcher(), profile="reservations")  # type: ignore[arg-type]
    create = next(
        tool for tool in tools if "create_reservation" in str(getattr(tool, "__name__", tool))
    )
    hints = get_type_hints(create)
    assert hints["context"] is RunContext


@pytest.mark.asyncio
async def test_livekit_tools_reservation_profile_hides_orders():
    pytest.importorskip("livekit.agents")
    from livekit_agent.tools.livekit_tools import build_livekit_tools

    dispatcher = FakeDispatcher()
    tools = build_livekit_tools(dispatcher, profile="reservations")  # type: ignore[arg-type]
    names = [str(getattr(tool, "__name__", getattr(tool, "name", tool))) for tool in tools]
    assert any("get_restaurant_info" in name for name in names)
    assert any("create_reservation" in name for name in names)
    assert any("check_table_availability" in name for name in names)
    assert any("list_available_slots" in name for name in names)
    assert not any("create_order" in name for name in names)
    assert not any("get_menu" in name for name in names)
