from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import Settings, apply_huggingface_token, get_settings
from app.node_client import NodeApiClient, NodeApiError
from app.runtime import VoiceRuntime
from app.session.manager import SessionManager
from app.telemetry import VoiceTelemetry
from app.tools.dispatcher import ToolDispatcher

logger = logging.getLogger("mysmartfood-voice")


class AvailabilityQuery(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")


class AppointmentPayload(BaseModel):
    name: str | None = None
    telephone: str
    date: str
    time: str
    type: str
    description: str | None = ""
    nombrePersonnes: int | None = 1
    commandes: list[dict[str, Any]] | None = None
    notes_internes: str | None = ""


def create_app(
    settings: Settings | None = None,
    runtime: VoiceRuntime | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        apply_huggingface_token(settings)
        node = runtime.node if runtime else NodeApiClient(settings)
        voice = runtime or _build_runtime(settings, node)
        app.state.settings = settings
        app.state.node = node
        app.state.runtime = voice
        app.state.sessions = SessionManager(voice)
        await _log_vllm_model(voice, settings)
        try:
            yield
        finally:
            await app.state.sessions.close_all()
            if runtime is None:
                await node.aclose()
            llm = voice.llm
            close = getattr(llm, "aclose", None)
            if runtime is None and close:
                await close()

    app = FastAPI(
        title="MySmartFood Voice",
        description="Runtime vocal local. Le metier restaurant reste dans Fastify.",
        lifespan=lifespan,
    )

    async def health_snapshot(
        include_active_sessions: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        node = app.state.node
        voice: VoiceRuntime = app.state.runtime
        node_started = time.perf_counter()
        try:
            ping = await node.ping()
            node_ok = not isinstance(ping, dict) or ping.get("success") is not False
        except Exception:
            ping = {"error": "Node indisponible"}
            node_ok = False
            if voice.telemetry:
                voice.telemetry.record_error("node")
        finally:
            if voice.telemetry:
                voice.telemetry.record_latency(
                    "node",
                    int((time.perf_counter() - node_started) * 1000),
                )
        engine_states = voice.engine_states()
        engines_ok = all(component["ready"] for component in engine_states.values())
        status = "ok" if node_ok and engines_ok else "degraded"
        sessions = getattr(app.state, "sessions", None)
        active_sessions = sessions.sanitized_sessions() if sessions is not None else []
        telemetry_snapshot = voice.telemetry.snapshot() if voice.telemetry else {}
        latency_by_component = telemetry_snapshot.get("latency_ms", {})
        errors_by_component = telemetry_snapshot.get("errors", {})
        components = {
            "node": {
                "ready": node_ok,
                "status": "healthy" if node_ok else "unhealthy",
                "error": None if node_ok else "unavailable",
                "latencyMs": latency_by_component.get("node", {}).get("last"),
                "errorCount": errors_by_component.get("node", 0),
            },
            **{
                name: {
                    "ready": state["ready"],
                    "status": "healthy" if state["ready"] else "unhealthy",
                    "error": None if state["ready"] else f"{name}_unavailable",
                    "latencyMs": latency_by_component.get(name, {}).get("last"),
                    "errorCount": errors_by_component.get(name, 0),
                }
                for name, state in engine_states.items()
            },
        }
        active_llm_provider = voice.active_llm_provider
        components["llm"]["provider"] = active_llm_provider
        components["llm"]["model"] = (
            settings.openai_model
            if active_llm_provider == "openai"
            else settings.vllm_model
        )
        payload: dict[str, Any] = {
            "status": status,
            "node": ping,
            "engines": {
                "ready": engines_ok,
                "error": None if engines_ok else "voice_engines_unavailable",
                "llm": settings.llm_provider,
                "llm_fallback_gpt": bool(settings.llm_fallback_gpt),
            },
            "components": components,
            "llm_provider_active": active_llm_provider,
            "sessions_active": len(active_sessions),
            "telemetry": telemetry_snapshot,
        }
        if include_active_sessions:
            payload["active_sessions"] = active_sessions
        return payload, node_ok and engines_ok

    @app.get("/live")
    async def live():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        payload, is_ready = await health_snapshot()
        return JSONResponse(payload, status_code=200 if is_ready else 503)

    @app.get("/health")
    async def health():
        payload, _ = await health_snapshot()
        return payload

    def require_internal_api_key(x_api_key: str | None = Header(default=None)) -> None:
        configured_key = settings.node_api_key.strip()
        provided_key = (x_api_key or "").strip()
        if (
            not configured_key
            or not provided_key
            or not secrets.compare_digest(configured_key, provided_key)
        ):
            raise HTTPException(status_code=401, detail="Clé API interne invalide")

    @app.get("/monitoring")
    async def monitoring(x_api_key: str | None = Header(default=None)):
        require_internal_api_key(x_api_key)
        payload, _ = await health_snapshot(include_active_sessions=True)
        return payload

    @app.post("/v1/tools/check-availability")
    async def check_availability(
        body: AvailabilityQuery,
        x_api_key: str | None = Header(default=None),
    ):
        require_internal_api_key(x_api_key)
        try:
            return await app.state.node.check_availability(body.date)
        except NodeApiError as exc:
            raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    @app.post("/v1/tools/create-appointment")
    async def create_appointment(
        body: AppointmentPayload,
        x_api_key: str | None = Header(default=None),
    ):
        require_internal_api_key(x_api_key)
        try:
            return await app.state.node.create_appointment(body.model_dump())
        except NodeApiError as exc:
            raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    @app.websocket("/media-stream")
    async def media_stream(websocket: WebSocket):
        await websocket.accept()
        client = websocket.client
        logger.info("step=twilio.ws.accept client=%s", client)
        await _run_native_media_stream(websocket, app.state.sessions)

    return app


async def _run_native_media_stream(websocket: WebSocket, manager: SessionManager) -> None:
    session = manager.create(websocket)
    close_reason = "disconnect"
    try:
        while True:
            message = await _receive_twilio_message(websocket)
            ended = await session.handle_raw(message)
            if ended:
                close_reason = "twilio.stop"
                break
    except WebSocketDisconnect:
        close_reason = "ws.disconnect"
        logger.info("step=twilio.ws.disconnect stream=%s call=%s", session.stream_sid, session.call_sid)
    except Exception:
        close_reason = "error"
        logger.exception(
            "step=twilio.ws.error stream=%s call=%s",
            session.stream_sid,
            session.call_sid,
        )
    finally:
        await manager.close(websocket, reason=close_reason)
        logger.info(
            "step=twilio.ws.cleanup stream=%s call=%s reason=%s active=%s",
            session.stream_sid,
            session.call_sid,
            close_reason,
            manager.active_count,
        )


async def _receive_twilio_message(websocket: WebSocket) -> str | bytes:
    """Fastify proxifie souvent les messages Twilio en frames binaires UTF-8."""
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code") or 1000)
    text = message.get("text")
    if text is not None:
        return text
    payload = message.get("bytes")
    if payload is not None:
        return payload
    raise WebSocketDisconnect(1000)


async def _log_vllm_model(voice: VoiceRuntime, settings: Settings) -> None:
    list_models = getattr(voice.llm, "list_models", None)
    if not callable(list_models):
        return
    try:
        served = await list_models()
    except Exception as exc:
        has_gpt = getattr(voice.llm, "_fallback", None) is not None
        if has_gpt:
            logger.warning(
                "FALLBACK GPT: vLLM injoignable au demarrage reason=%s. "
                "Completions via OpenAI Chat Completions.",
                exc,
            )
            return
        logger.warning(
            "FALLBACK GPT: vLLM injoignable et fallback LLM inactif (OPENAI_API_KEY vide). "
            "Fastify pourra basculer l appel vers OpenAI Realtime. reason=%s",
            exc,
        )
        voice.engines_ready = False
        voice.engine_error = "vLLM injoignable, fallback LLM inactif"
        if voice.component_errors is not None:
            voice.component_errors["llm"] = voice.engine_error
        return
    from app.llm.vllm import model_id_error

    mismatch = model_id_error(settings.vllm_model, served)
    if mismatch:
        logger.error("step=vllm.model_mismatch %s", mismatch)
        voice.engines_ready = False
        voice.engine_error = "Modele vLLM configure indisponible"
        if voice.component_errors is not None:
            voice.component_errors["llm"] = voice.engine_error
    else:
        logger.info("step=vllm.model_ok model=%s served=%s", settings.vllm_model, served)


def _build_runtime(settings: Settings, node: NodeApiClient) -> VoiceRuntime:
    apply_huggingface_token(settings)
    dispatcher = ToolDispatcher(node)
    telemetry = VoiceTelemetry()
    if not settings.voice_eager_load:
        return VoiceRuntime(
            settings=settings,
            node=node,
            dispatcher=dispatcher,
            llm=None,
            stt=None,
            tts=None,
            vad_model=None,
            engines_ready=False,
            engine_error="VOICE_EAGER_LOAD=false (mode test / demarrage leger)",
            telemetry=telemetry,
        )
    try:
        from app.llm.fallback import build_llm
        from app.providers import build_engines

        bundle = build_engines(settings, eager=True)
        llm = build_llm(settings, telemetry)
        logger.info(
            "moteurs vocaux precharges stt=%s tts=%s llm=%s turn=%s orchestrator=%s",
            bundle.stt_provider,
            bundle.tts_provider,
            bundle.llm_provider,
            bundle.turn_provider,
            bundle.orchestrator,
        )
        return VoiceRuntime(
            settings=settings,
            node=node,
            dispatcher=dispatcher,
            llm=llm,
            stt=bundle.stt,
            tts=bundle.tts,
            vad_model=bundle.vad_model,
            engines_ready=True,
            telemetry=telemetry,
            turn=bundle.turn,
            stt_provider=bundle.stt_provider,
            tts_provider=bundle.tts_provider,
            llm_provider_name=bundle.llm_provider,
        )
    except Exception as exc:
        logger.exception("chargement moteurs vocal echoue")
        return VoiceRuntime(
            settings=settings,
            node=node,
            dispatcher=dispatcher,
            llm=None,
            stt=None,
            tts=None,
            vad_model=None,
            engines_ready=False,
            engine_error=str(exc),
            telemetry=telemetry,
        )


app = create_app()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.voice_host,
        port=settings.voice_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
