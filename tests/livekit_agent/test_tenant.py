from dataclasses import replace

import httpx
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.session import CallIdentity
from livekit_agent.tenant import resolve_session_identity
from livekit_agent.tools.http import FastifyToolsClient


def _settings(**overrides) -> LiveKitSettings:
    values = {
        "_env_file": None,
        "node_base_url": "http://fastify.test",
        "node_api_key": "secret",
    }
    values.update(overrides)
    return LiveKitSettings(**values)


@pytest.mark.asyncio
async def test_resolve_fastify_wins_over_preset_id_on_sip_call():
    identity = CallIdentity(
        call_id="CA1",
        session_id="call-1",
        restaurant_id="tenant-mapped",
        trunk_number="+33111111111",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["phone"] == "+33111111111"
        return httpx.Response(
            200,
            json={"success": True, "data": {"restaurantId": "tenant-42", "slug": "le-comptoir", "status": "active"}},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fastify.test")
    client = FastifyToolsClient(_settings(), client=http)
    resolved = await resolve_session_identity(_settings(), identity, client=client)
    assert resolved.restaurant_id == "tenant-42"
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_looks_up_backoffice_did():
    identity = CallIdentity(
        call_id="CA1",
        session_id="call-1",
        restaurant_id="",
        trunk_number="+33111111111",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["phone"] == "+33111111111"
        return httpx.Response(
            200,
            json={"success": True, "data": {"restaurantId": "tenant-42", "slug": "le-comptoir", "status": "active"}},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fastify.test")
    client = FastifyToolsClient(_settings(), client=http)
    resolved = await resolve_session_identity(_settings(), identity, client=client)
    assert resolved == replace(identity, restaurant_id="tenant-42")
    await client.aclose()


@pytest.mark.asyncio
async def test_unknown_backoffice_did_stays_unresolved():
    identity = CallIdentity(
        call_id="CA1",
        session_id="call-1",
        restaurant_id="",
        trunk_number="+33999999999",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"success": False, "error": "Instance non trouvée"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fastify.test")
    client = FastifyToolsClient(_settings(), client=http)
    resolved = await resolve_session_identity(_settings(), identity, client=client)
    assert resolved.restaurant_id == ""
    await client.aclose()
