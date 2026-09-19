from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from livekit_agent.config import LiveKitSettings
from livekit_agent.llm.client import build_llm
from livekit_agent.observability.logging import emit_event
from livekit_agent.observability.metrics import CallObserver
from livekit_agent.observability.resources import SESSIONS
from livekit_agent.prompts import build_phase4_instructions, build_tool_instructions
from livekit_agent.restaurant_agent import build_restaurant_agent
from livekit_agent.session import CallIdentity, extract_call_identity, first_sip_attributes
from livekit_agent.stt.whisper import build_whisper_stt
from livekit_agent.tools.dispatcher import ToolDispatcher
from livekit_agent.tools.http import FastifyToolsClient
from livekit_agent.tts.kokoro import build_kokoro_tts


@dataclass(frozen=True)
class SessionPlan:
    dispatcher: ToolDispatcher | None
    instructions: str
    greeting: str

    @property
    def uses_tools(self) -> bool:
        return self.dispatcher is not None


async def _sip_attributes(ctx: Any) -> dict[str, Any]:
    wait_for = getattr(ctx, "wait_for_participant", None)
    if callable(wait_for):
        try:
            participant = await wait_for()
            attributes = getattr(participant, "attributes", None) or {}
            if attributes:
                return dict(attributes)
        except Exception:
            pass
    return first_sip_attributes(getattr(ctx, "room", None))


def bind_call_context(ctx: Any, settings: LiveKitSettings, sip_attributes: dict[str, Any]) -> CallIdentity:
    room = getattr(ctx, "room", None)
    room_name = getattr(room, "name", "") or "unknown-room"
    identity = extract_call_identity(
        settings,
        room_name=room_name,
        sip_attributes=sip_attributes,
    )
    ctx.log_context_fields = {
        "room": room_name,
        "call_id": identity.call_id,
        "session_id": identity.session_id,
        "restaurant_id": identity.restaurant_id,
    }
    return identity


def plan_session(settings: LiveKitSettings, restaurant_id: str) -> SessionPlan:
    dispatcher = None
    if settings.voice_tools_enabled:
        dispatcher = ToolDispatcher(FastifyToolsClient(settings), restaurant_id)
    instructions = (
        build_tool_instructions(restaurant_name=settings.restaurant_name)
        if dispatcher is not None
        else build_phase4_instructions(restaurant_name=settings.restaurant_name)
    )
    return SessionPlan(
        dispatcher=dispatcher,
        instructions=instructions,
        greeting=settings.greeting_text,
    )


def _load_pipeline(settings: LiveKitSettings):
    from livekit.agents import AgentSession, stt
    from livekit.plugins import silero

    vad = silero.VAD.load()
    return AgentSession(
        stt=stt.StreamAdapter(stt=build_whisper_stt(settings), vad=vad),
        llm=build_llm(settings),
        tts=build_kokoro_tts(settings),
        vad=vad,
        allow_interruptions=True,
    )


async def start_voice_session(ctx: Any, settings: LiveKitSettings) -> None:
    identity = bind_call_context(ctx, settings, await _sip_attributes(ctx))
    observer = CallObserver(identity)
    concurrent = SESSIONS.register(identity.session_id)
    emit_event("session.start", concurrent_sessions=concurrent, **observer._ids())

    plan = plan_session(settings, identity.restaurant_id)
    dispatcher = plan.dispatcher
    agent = build_restaurant_agent(
        instructions=plan.instructions,
        greeting=plan.greeting,
        dispatcher=dispatcher,
    )
    session = _load_pipeline(settings)

    def _on_metrics(ev: Any) -> None:
        observer.on_livekit_metrics(getattr(ev, "metrics", ev))

    def _on_transcript(ev: Any) -> None:
        text = getattr(ev, "transcript", None) or getattr(ev, "text", "") or ""
        is_final = bool(getattr(ev, "is_final", True))
        observer.on_user_transcript(str(text), is_final=is_final)

    session.on("metrics_collected", _on_metrics)
    session.on("user_input_transcribed", _on_transcript)

    try:
        await session.start(room=getattr(ctx, "room", None), agent=agent)
    finally:
        observer.on_session_end()
        SESSIONS.release(identity.session_id)
        if dispatcher is not None:
            await dispatcher.aclose()
