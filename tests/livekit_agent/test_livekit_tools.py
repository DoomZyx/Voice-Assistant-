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
