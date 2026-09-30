from types import SimpleNamespace

import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.observability.resources import SESSIONS
from livekit_agent.prompts import build_phase4_instructions
from livekit_agent.runtime import (
    _load_restaurant_name,
    _sip_attributes,
    bind_call_context,
    plan_session,
    start_voice_session,
)


def _settings(**overrides) -> LiveKitSettings:
    values = {
        "_env_file": None,
        "restaurant_name": "Le Comptoir",
        "greeting_text": "Bonjour, je vous ecoute.",
        "voice_tools_enabled": False,
    }
    values.update(overrides)
    return LiveKitSettings(**values)


def test_phase4_plan_has_no_fastify_dispatcher():
    plan = plan_session(_settings(), "tenant-1")
    assert plan.uses_tools is False
    assert plan.dispatcher is None
    assert plan.greeting == "Le Comptoir bonjour."
    assert plan.instructions == build_phase4_instructions(restaurant_name="Le Comptoir")
    assert "get_restaurant_info" not in plan.instructions


def test_tool_plan_wires_dispatcher():
    plan = plan_session(_settings(voice_tools_enabled=True), "tenant-1")
    assert plan.uses_tools is True
    assert plan.dispatcher is not None
    assert "get_restaurant_info" in plan.instructions


def test_reservation_profile_uses_reservation_prompt():
    plan = plan_session(
        _settings(voice_tools_enabled=True, voice_tools_profile="reservations"),
        "tenant-1",
    )
    assert plan.uses_tools is True
    assert plan.dispatcher is not None
    assert plan.dispatcher.profile == "reservations"
    assert "create_reservation" in plan.instructions
    assert "create_order" not in plan.instructions


def test_full_profile_uses_both_business_flows_and_tenant_name():
    plan = plan_session(
        _settings(voice_tools_enabled=True, voice_tools_profile="full"),
        "tenant-2",
        restaurant_name="Chez Alice",
    )
    assert plan.dispatcher is not None
    assert plan.dispatcher.profile == "full"
    assert plan.greeting == "Chez Alice bonjour."
    assert "Chez Alice" in plan.instructions
    assert "create_reservation" in plan.instructions
    assert "create_order" in plan.instructions
    assert "chaises hautes" in plan.instructions


@pytest.mark.asyncio
async def test_load_restaurant_name_uses_fastify_tenant_context(monkeypatch):
    import httpx

    from livekit_agent.tools.http import FastifyToolsClient

    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("x-tenant-id")))
        return httpx.Response(200, json={"data": {"restaurantInfo": {"nom": "Chez Alice"}}})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://fastify.test"
    )
    monkeypatch.setattr(
        "livekit_agent.runtime.FastifyToolsClient",
        lambda settings: FastifyToolsClient(settings, client=client),
    )
    assert await _load_restaurant_name(_settings(), "tenant-2") == "Chez Alice"
    assert seen == [("/api/voice/context/tenant-2", "tenant-2")]
    await client.aclose()


@pytest.mark.asyncio
async def test_start_voice_session_rejects_missing_restaurant_name(monkeypatch):
    async def missing_name(_settings, restaurant_id):
        assert restaurant_id == "tenant-1"
        return ""

    def must_not_start(_settings):
        raise AssertionError("Audio session must not start without restaurant name")

    monkeypatch.setattr("livekit_agent.runtime._load_restaurant_name", missing_name)
    monkeypatch.setattr("livekit_agent.runtime._load_pipeline", must_not_start)
    ctx = SimpleNamespace(room=SimpleNamespace(name="call-no-name"), log_context_fields=None)
    SESSIONS.release("call-no-name")
    with pytest.raises(RuntimeError, match="Restaurant name unavailable"):
        await start_voice_session(
            ctx,
            _settings(restaurant_id="tenant-1", voice_tools_enabled=True, voice_tools_profile="full"),
        )
    assert "call-no-name" not in SESSIONS._ids


def test_bind_call_context_sets_structured_ids():
    ctx = SimpleNamespace(room=SimpleNamespace(name="call-abc"), log_context_fields=None)
    identity = bind_call_context(
        ctx,
        _settings(restaurant_id="tenant-1"),
        {"sip.callID": "CA123"},
    )
    assert identity.call_id == "CA123"
    assert identity.session_id == "call-abc"
    assert ctx.log_context_fields == {
        "room": "call-abc",
        "call_id": "CA123",
        "session_id": "call-abc",
        "restaurant_id": "tenant-1",
        "tenant_id": "tenant-1",
        "instance_id": "tenant-1",
    }


@pytest.mark.asyncio
async def test_sip_attributes_prefer_wait_for_participant():
    async def wait_for():
        return SimpleNamespace(attributes={"sip.callId": "SIP-9"})

    ctx = SimpleNamespace(
        wait_for_participant=wait_for,
        room=SimpleNamespace(remote_participants={}),
    )
    assert await _sip_attributes(ctx) == {"sip.callId": "SIP-9"}


@pytest.mark.asyncio
async def test_sip_attributes_fall_back_when_wait_fails():
    async def boom():
        raise RuntimeError("gone")

    participant = SimpleNamespace(attributes={"sip.from": "+33600000000"})
    ctx = SimpleNamespace(
        wait_for_participant=boom,
        room=SimpleNamespace(remote_participants={"p1": participant}),
    )
    assert await _sip_attributes(ctx) == {"sip.from": "+33600000000"}


@pytest.mark.asyncio
async def test_start_voice_session_reuses_bound_identity(monkeypatch):
    from livekit_agent.session import CallIdentity, bind_job_identity

    captured: dict = {}

    class FakeSession:
        def on(self, name, fn) -> None:
            return None

        async def start(self, room, agent) -> None:
            captured["started"] = True

    async def boom(*_args, **_kwargs):
        raise AssertionError("DID lookup must not run again")

    monkeypatch.setattr("livekit_agent.runtime._load_pipeline", lambda settings: FakeSession())
    monkeypatch.setattr(
        "livekit_agent.runtime.build_restaurant_agent",
        lambda **kwargs: captured.setdefault("agent_kwargs", kwargs) or "agent",
    )
    monkeypatch.setattr("livekit_agent.runtime.resolve_session_identity", boom)

    room = SimpleNamespace(name="call-reuse")
    ctx = SimpleNamespace(room=room, log_context_fields=None)
    bound = CallIdentity(call_id="CA-bound", session_id="call-reuse", restaurant_id="tenant-bound")
    bind_job_identity(ctx, bound)
    SESSIONS.release("call-reuse")
    await start_voice_session(
        ctx,
        _settings(restaurant_id="tenant-other"),
    )
    assert ctx.log_context_fields["restaurant_id"] == "tenant-bound"
    assert captured["started"] is True


@pytest.mark.asyncio
async def test_start_voice_session_releases_registry_without_tools(monkeypatch):
    captured: dict = {}

    class FakeSession:
        def __init__(self) -> None:
            self.handlers: dict = {}

        def on(self, name, fn) -> None:
            self.handlers[name] = fn

        async def start(self, room, agent) -> None:
            captured["room"] = room
            captured["agent"] = agent
            captured["handlers"] = dict(self.handlers)

    monkeypatch.setattr("livekit_agent.runtime._load_pipeline", lambda settings: FakeSession())
    monkeypatch.setattr(
        "livekit_agent.runtime.build_restaurant_agent",
        lambda **kwargs: captured.setdefault("agent_kwargs", kwargs) or "agent",
    )

    room = SimpleNamespace(name="call-teardown")
    ctx = SimpleNamespace(room=room, log_context_fields=None)
    SESSIONS.release("call-teardown")
    await start_voice_session(ctx, _settings(restaurant_id="tenant-1"))
    assert "call-teardown" not in SESSIONS._ids
    assert captured["room"] is room
    assert captured["agent_kwargs"]["dispatcher"] is None
    assert "metrics_collected" in captured["handlers"]
    assert ctx.log_context_fields["call_id"] == "call-teardown"


@pytest.mark.asyncio
async def test_start_voice_session_keeps_http_client_until_shutdown(monkeypatch):
    captured: dict = {}
    closed: list[str] = []

    class FakeSession:
        def on(self, name, fn) -> None:
            return None

        async def start(self, room, agent) -> None:
            captured["started"] = True

    class TrackingDispatcher:
        def __init__(self) -> None:
            self.profile = "reservations"

        async def aclose(self) -> None:
            closed.append("dispatcher")

    class FakePlan:
        dispatcher = TrackingDispatcher()
        instructions = "reserve"
        greeting = "bonjour"
        uses_tools = True

    monkeypatch.setattr("livekit_agent.runtime._load_pipeline", lambda settings: FakeSession())
    monkeypatch.setattr(
        "livekit_agent.runtime.build_restaurant_agent",
        lambda **kwargs: captured.setdefault("agent_kwargs", kwargs) or "agent",
    )
    monkeypatch.setattr("livekit_agent.runtime.plan_session", lambda *a, **k: FakePlan())
    async def tenant_name(_settings, restaurant_id):
        assert restaurant_id == "tenant-1"
        return "Chez Alice"

    monkeypatch.setattr("livekit_agent.runtime._load_restaurant_name", tenant_name)

    shutdown_cbs: list = []

    def add_shutdown_callback(cb) -> None:
        shutdown_cbs.append(cb)

    room = SimpleNamespace(name="call-keep-http")
    ctx = SimpleNamespace(
        room=room,
        log_context_fields=None,
        add_shutdown_callback=add_shutdown_callback,
    )
    SESSIONS.release("call-keep-http")
    await start_voice_session(
        ctx,
        _settings(restaurant_id="tenant-1", voice_tools_enabled=True, voice_tools_profile="reservations"),
    )
    assert captured["started"] is True
    assert closed == []
    assert "call-keep-http" in SESSIONS._ids
    assert len(shutdown_cbs) == 1
    await shutdown_cbs[0]("test")
    assert closed == ["dispatcher"]
    assert "call-keep-http" not in SESSIONS._ids
