import httpx
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.tools.dispatcher import ToolDispatcher
from livekit_agent.tools.http import FastifyToolsClient


@pytest.mark.asyncio
async def test_llm_tool_call_reaches_fastify_restaurant_info():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        assert request.headers.get("x-api-key") == "secret"
        assert request.headers.get("x-tenant-id") == "tenant-1"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "restaurantInfo": {"nom": "Le Comptoir", "adresse": "1 rue Test"},
                    "menu": {"burgers": {"nom": "Burgers", "produits": [{"nom": "X", "prix": 1}]}},
                },
            },
        )

    settings = LiveKitSettings(node_base_url="http://fastify.test", node_api_key="secret")
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://fastify.test",
        headers={"x-api-key": "secret", "Content-Type": "application/json"},
    )
    dispatcher = ToolDispatcher(FastifyToolsClient(settings, client=http), tenant_id="tenant-1")
    llm_tool_call = {"name": "get_restaurant_info", "arguments": {}}
    result = await dispatcher.dispatch(llm_tool_call["name"], llm_tool_call["arguments"])
    await dispatcher.aclose()
    assert result["success"] is True
    assert result["data"]["nom"] == "Le Comptoir"
    assert any(path.endswith("/api/voice/context/tenant-1") for path in seen)
