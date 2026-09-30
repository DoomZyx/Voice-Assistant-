from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from livekit_agent.config import LiveKitSettings
from livekit_agent.llm.client import build_llm
from livekit_agent.observability.audio_path import log_session_audio_contract
from livekit_agent.observability.logging import emit_event
from livekit_agent.observability.metrics import CallObserver
from livekit_agent.observability.resources import SESSIONS
from livekit_agent.prompts import (
    build_full_instructions,
    build_phase4_instructions,
    build_reservation_instructions,
    build_tool_instructions,
)
from livekit_agent.restaurant_agent import build_restaurant_agent
from livekit_agent.session import (
    CallIdentity,
    bind_job_identity,
    extract_call_identity,
    first_sip_attributes,
    read_job_identity,
)
from livekit_agent.tenant import resolve_session_identity
from livekit_agent.stt.whisper import build_whisper_stt
from livekit_agent.tools.dispatcher import ToolDispatcher
from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient
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
            sip_only = {
                str(key): value
                for key, value in dict(attributes).items()
                if str(key).startswith("sip.")
            }
            if sip_only:
                return sip_only
        except Exception:
            pass
    return first_sip_attributes(getattr(ctx, "room", None))


def bind_call_context(ctx: Any, settings: LiveKitSettings, sip_attributes: dict[str, Any]) -> CallIdentity:
    existing = read_job_identity(ctx)
    if existing is not None:
        return existing
    room = getattr(ctx, "room", None)
    room_name = getattr(room, "name", "") or "unknown-room"
    identity = extract_call_identity(
        settings,
        room_name=room_name,
        sip_attributes=sip_attributes,
    )
    return bind_job_identity(ctx, identity)


def plan_session(
    settings: LiveKitSettings,
    restaurant_id: str,
    session_store: dict[str, Any] | None = None,
    restaurant_name: str | None = None,
) -> SessionPlan:
    profile = (settings.voice_tools_profile or "").strip().lower()
    name = settings.restaurant_name if restaurant_name is None else restaurant_name
    name = name.strip()
    dispatcher = None
    if settings.voice_tools_enabled:
        dispatcher = ToolDispatcher(
            FastifyToolsClient(settings),
            restaurant_id,
            session_store=session_store,
            profile=profile,
        )
    if dispatcher is None:
        instructions = build_phase4_instructions(restaurant_name=name)
    elif profile == "reservations":
        instructions = build_reservation_instructions(restaurant_name=name)
    elif profile == "full":
        instructions = build_full_instructions(restaurant_name=name)
    else:
        instructions = build_tool_instructions(restaurant_name=name)
    return SessionPlan(
        dispatcher=dispatcher,
        instructions=instructions,
        greeting=f"{name} bonjour." if name else settings.greeting_text,
    )


async def _load_restaurant_name(settings: LiveKitSettings, restaurant_id: str) -> str:
    if not restaurant_id:
        return ""
    client = FastifyToolsClient(settings)
    try:
        context = await client.get_voice_context(restaurant_id)
        info = context.get("restaurantInfo")
        return str(info.get("nom") or "").strip() if isinstance(info, dict) else ""
    except (FastifyApiError, httpx.HTTPError) as exc:
        emit_event("restaurant.context_unavailable", restaurant_id=restaurant_id, reason=str(exc))
        return ""
    finally:
        await client.aclose()


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
    sip_attributes = await _sip_attributes(ctx)
    existing = read_job_identity(ctx)
    if existing is None:
        room = getattr(ctx, "room", None)
        extracted = extract_call_identity(
            settings,
            room_name=getattr(room, "name", "") or "unknown-room",
            sip_attributes=sip_attributes,
        )
        identity = bind_job_identity(ctx, await resolve_session_identity(settings, extracted))
    else:
        identity = existing
    restaurant_name = None
    if settings.voice_tools_enabled:
        restaurant_name = await _load_restaurant_name(settings, identity.restaurant_id)
        if not restaurant_name:
            emit_event("restaurant.name_unavailable", restaurant_id=identity.restaurant_id)
            raise RuntimeError("Restaurant name unavailable for voice greeting")
    observer = CallObserver(identity)
    concurrent = SESSIONS.register(identity.session_id)
    emit_event("session.start", concurrent_sessions=concurrent, **observer._ids())
    log_session_audio_contract(
        identity,
        sip_attributes=sip_attributes,
        kokoro_hz=settings.tts_sample_rate,
    )

    userdata = getattr(ctx, "userdata", None)
    if not isinstance(userdata, dict):
        userdata = {}
        ctx.userdata = userdata
    if identity.caller_number and not userdata.get("caller_phone"):
        userdata["caller_phone"] = identity.caller_number
    plan = plan_session(
        settings,
        identity.restaurant_id,
        session_store=userdata,
        restaurant_name=restaurant_name,
    )
    dispatcher = plan.dispatcher
    agent = build_restaurant_agent(
        instructions=plan.instructions,
        greeting=plan.greeting,
        dispatcher=dispatcher,
        tools_profile=(settings.voice_tools_profile or "").strip().lower(),
    )
    session = _load_pipeline(settings)

    def _on_metrics(ev: Any) -> None:
        observer.on_livekit_metrics(getattr(ev, "metrics", ev))

    def _on_transcript(ev: Any) -> None:
        text = getattr(ev, "transcript", None) or getattr(ev, "text", "") or ""
        is_final = bool(getattr(ev, "is_final", True))
        observer.on_user_transcript(str(text), is_final=is_final)

    def _on_agent_state(ev: Any) -> None:
        emit_event(
            "turn.agent_state",
            state=getattr(ev, "new_state", None) or getattr(ev, "state", None),
            **observer._ids(),
        )

    def _on_user_state(ev: Any) -> None:
        emit_event(
            "turn.user_state",
            state=getattr(ev, "new_state", None) or getattr(ev, "state", None),
            **observer._ids(),
        )

    for event_name, handler in (
        ("metrics_collected", _on_metrics),
        ("user_input_transcribed", _on_transcript),
        ("agent_state_changed", _on_agent_state),
        ("user_state_changed", _on_user_state),
    ):
        try:
            session.on(event_name, handler)
        except Exception:
            emit_event("turn.hook_missing", hook=event_name, **observer._ids())

    # session.start() returns after connect/setup, not when the PSTN call ends.
    # Closing Fastify here killed tools mid-call (greeting OK, then silence).
    cleaned = False

    async def _cleanup(_reason: str = "") -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        observer.on_session_end()
        SESSIONS.release(identity.session_id)
        if dispatcher is not None:
            await dispatcher.aclose()

    add_shutdown = getattr(ctx, "add_shutdown_callback", None)
    if callable(add_shutdown):
        add_shutdown(_cleanup)
        await session.start(room=getattr(ctx, "room", None), agent=agent)
        return

    try:
        await session.start(room=getattr(ctx, "room", None), agent=agent)
    finally:
        await _cleanup()
