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


def test_health_degraded_without_engines():
    settings = Settings(
        voice_eager_load=False,
        node_base_url="http://node.test",
        node_api_key="test-key",
        llm_fallback_gpt=True,
    )
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
        live = client.get("/live")
        ready = client.get("/ready")
        monitoring_denied = client.get("/monitoring")
        monitoring_allowed = client.get(
            "/monitoring",
            headers={"x-api-key": settings.node_api_key},
        )
    assert response.status_code == 200
    assert live.status_code == 200
    assert ready.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["node"]["success"] is True
    assert body["engines"]["llm_fallback_gpt"] is True
    assert body["sessions_active"] == 0
    assert body["components"]["node"]["ready"] is True
    assert body["components"]["vad"]["ready"] is False
    assert "active_sessions" not in body
    assert "latency_ms" in body["telemetry"]
    assert monitoring_denied.status_code == 401
    assert monitoring_allowed.status_code == 200
    assert monitoring_allowed.json()["active_sessions"] == []


def test_ready_and_enriched_health_with_all_components():
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
    runtime.telemetry.record_latency("stt", 12)
    runtime.telemetry.record_error("tts")
    app = create_app(settings, runtime)

    with TestClient(app) as client:
        ready = client.get("/ready")
        health = client.get("/health")

    assert ready.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert all(component["ready"] for component in body["components"].values())
    assert body["llm_provider_active"] == "vllm"
    assert body["components"]["llm"]["status"] == "healthy"
    assert body["components"]["llm"]["provider"] == "vllm"
    assert body["components"]["llm"]["model"] == settings.vllm_model
    assert body["telemetry"]["latency_ms"]["stt"]["last"] == 12
    assert body["telemetry"]["errors"]["tts"] == 1
