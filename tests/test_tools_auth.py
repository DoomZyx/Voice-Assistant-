import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.node_client import NodeApiClient
from app.runtime import VoiceRuntime
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeLLM, FakeSTT, FakeTTS, FakeVadModel


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/ping":
        return httpx.Response(200, json={"success": True, "message": "Backend actif"})
    return httpx.Response(404, json={"error": "not found"})


def _app():
    settings = Settings(
        voice_eager_load=False,
        node_base_url="http://node.test",
        node_api_key="test-key",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(_handler), base_url="http://node.test")
    node = NodeApiClient(settings, client=http)
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=FakeLLM(),
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad_model=FakeVadModel(),
        engines_ready=True,
    )
    return create_app(settings, runtime)


def test_voice_tools_require_internal_key():
    with TestClient(_app()) as client:
        availability = client.post("/v1/tools/check-availability", json={"date": "2026-09-19"})
        appointment = client.post(
            "/v1/tools/create-appointment",
            json={
                "telephone": "0600000000",
                "date": "2026-09-19",
                "time": "19:00",
                "type": "Réservation de table",
            },
        )

    assert availability.status_code == 401
    assert appointment.status_code == 401
