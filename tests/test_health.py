import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.node_client import NodeApiClient
from app.runtime import VoiceRuntime
from app.tools.dispatcher import ToolDispatcher


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/ping":
        return httpx.Response(200, json={"success": True, "message": "Backend actif"})
    return httpx.Response(404, json={"error": "not found"})


def test_health_degraded_without_engines():
    settings = Settings(voice_eager_load=False, node_base_url="http://node.test")
    http = httpx.AsyncClient(transport=httpx.MockTransport(_handler), base_url="http://node.test")
    node = NodeApiClient(settings, client=http)
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=None,
        stt=None,
        tts=None,
        vad_model=None,
        engines_ready=False,
        engine_error="test",
    )
    app = create_app(settings, runtime)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["node"]["success"] is True
    assert body["engines"]["llm_fallback_gpt"] is True
    assert body["sessions_active"] == 0
