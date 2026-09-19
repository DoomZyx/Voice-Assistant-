from types import SimpleNamespace

import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.observability.resources import SESSIONS
from livekit_agent.prompts import build_phase4_instructions
from livekit_agent.runtime import (
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
    assert plan.greeting == "Bonjour, je vous ecoute."
    assert plan.instructions == build_phase4_instructions(restaurant_name="Le Comptoir")
    assert "get_restaurant_info" not in plan.instructions


def test_tool_plan_wires_dispatcher():
    plan = plan_session(_settings(voice_tools_enabled=True), "tenant-1")
    assert plan.uses_tools is True
    assert plan.dispatcher is not None
    assert "get_restaurant_info" in plan.instructions


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
