from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.config import Settings, apply_huggingface_token, get_settings
from app.node_client import NodeApiClient, NodeApiError
from app.runtime import VoiceRuntime
from app.session.manager import SessionManager
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

    @app.get("/health")
    async def health():
        node = app.state.node
        voice: VoiceRuntime = app.state.runtime
        try:
            ping = await node.ping()
            node_ok = True
        except Exception as exc:
            ping = {"error": str(exc)}
            node_ok = False
        engines_ok = voice.engines_ready
        status = "ok" if node_ok and engines_ok else "degraded"
        sessions = getattr(app.state, "sessions", None)
        return {
            "status": status,
            "node": ping,
            "engines": {
                "ready": engines_ok,
                "error": voice.engine_error,
                "llm": settings.llm_provider,
            },
            "sessions_active": sessions.active_count if sessions is not None else 0,
        }

    @app.post("/v1/tools/check-availability")
    async def check_availability(body: AvailabilityQuery):
        try:
            return await app.state.node.check_availability(body.date)
        except NodeApiError as exc:
            raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    @app.post("/v1/tools/create-appointment")
    async def create_appointment(body: AppointmentPayload):
        try:
            return await app.state.node.create_appointment(body.model_dump())
        except NodeApiError as exc:
            raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    @app.websocket("/media-stream")
    async def media_stream(websocket: WebSocket):
        await websocket.accept()
        client = websocket.client
        logger.info("step=twilio.ws.accept client=%s", client)
        manager: SessionManager = app.state.sessions
        session = manager.create(websocket)
        close_reason = "disconnect"
        try:
            while True:
                message = await websocket.receive_text()
                ended = await session.handle_raw(message)
                if ended:
                    close_reason = "twilio.stop"
                    break
        except WebSocketDisconnect:
            close_reason = "ws.disconnect"
            logger.info("step=twilio.ws.disconnect stream=%s call=%s", session.stream_sid, session.call_sid)
        except Exception:
            close_reason = "error"
            logger.exception("step=twilio.ws.error stream=%s call=%s", session.stream_sid, session.call_sid)
        finally:
            await manager.close(websocket, reason=close_reason)
            logger.info(
                "step=twilio.ws.cleanup stream=%s call=%s reason=%s active=%s",
                session.stream_sid,
                session.call_sid,
                close_reason,
                manager.active_count,
            )

    return app


async def _log_vllm_model(voice: VoiceRuntime, settings: Settings) -> None:
    list_models = getattr(voice.llm, "list_models", None)
    if not callable(list_models):
        return
    try:
        served = await list_models()
    except Exception as exc:
        logger.warning("step=vllm.models_unreachable error=%s", exc)
        return
    from app.llm.vllm import model_id_error

    mismatch = model_id_error(settings.vllm_model, served)
    if mismatch:
        logger.error("step=vllm.model_mismatch %s", mismatch)
    else:
        logger.info("step=vllm.model_ok model=%s served=%s", settings.vllm_model, served)


def _build_runtime(settings: Settings, node: NodeApiClient) -> VoiceRuntime:
    apply_huggingface_token(settings)
    dispatcher = ToolDispatcher(node)
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
        )
    try:
        from app.llm.vllm import build_llm
        from app.stt.faster_whisper import FasterWhisperSTT
        from app.tts.kokoro import KokoroTTS
        from app.vad.silero import load_silero_model

        vad_model = load_silero_model(settings)
        stt = FasterWhisperSTT(settings)
        tts = KokoroTTS(settings)
        llm = build_llm(settings)
        logger.info("moteurs vocaux precharges")
        return VoiceRuntime(
            settings=settings,
            node=node,
            dispatcher=dispatcher,
            llm=llm,
            stt=stt,
            tts=tts,
            vad_model=vad_model,
            engines_ready=True,
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
