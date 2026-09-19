"""Appel Twilio Media Streams de bout en bout contre /media-stream uniquement."""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi.testclient import TestClient

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.llm import LLMResult
from app.main import create_app
from app.runtime import VoiceRuntime
from app.stream_token import create_stream_token
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeLLM, FakeNode, FakeSTT, FakeTTS, FakeVadModel

STREAM_SID = "MZ_E2E"
CALL_SID = "CA_E2E"
GREETING_FRAMES = 3
STREAM_SECRET = "test-twilio-token"
INSTANCE_ID = "restaurant-e2e"


def _media_msg(samples: int = 2000) -> str:
    import base64

    pcm = np.zeros(samples, dtype=np.int16)
    payload = base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")
    return json.dumps(
        {
            "event": "media",
            "streamSid": STREAM_SID,
            "media": {"payload": payload, "track": "inbound", "chunk": "1", "timestamp": "0"},
        }
    )


def _start_msg() -> str:
    return json.dumps(
        {
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "streamSid": STREAM_SID,
                "callSid": CALL_SID,
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": {
                    "callerNumber": "+33612345678",
                    "instanceId": INSTANCE_ID,
                    "streamToken": create_stream_token(INSTANCE_ID, STREAM_SECRET),
                },
            },
        }
    )


def _build_app(stt=None, llm=None, tts=None):
    settings = Settings(
        voice_eager_load=False,
        greeting_text="Bonjour, je vous ecoute.",
        node_api_key="test-key",
        twilio_auth_token=STREAM_SECRET,
    )
    node = FakeNode()
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=llm or FakeLLM(LLMResult(content="Pour quelle date ?")),
        stt=stt or FakeSTT("une table pour deux"),
        tts=tts or FakeTTS(),
        vad_model=FakeVadModel(end_on=2),
        engines_ready=True,
    )
    return create_app(settings, runtime), runtime


def _receive_media(ws, count: int) -> list[dict]:
    frames = []
    for _ in range(count):
        message = ws.receive_json()
        assert message["event"] == "media"
        assert message["streamSid"] == STREAM_SID
        assert message["media"]["payload"]
        frames.append(message)
    return frames


def test_twilio_call_e2e_stop_cleans_session(caplog):
    caplog.set_level(logging.INFO)
    app, runtime = _build_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["sessions_active"] == 0
        with client.websocket_connect("/media-stream") as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            ws.send_bytes(_start_msg().encode("utf-8"))
            greeting = _receive_media(ws, GREETING_FRAMES)
            assert greeting
            ws.send_text(_media_msg())
            ws.send_text(_media_msg())
            reply = _receive_media(ws, GREETING_FRAMES)
            assert reply
            ws.send_text(json.dumps({"event": "stop", "streamSid": STREAM_SID}))
        assert app.state.sessions.active_count == 0
        assert client.get("/health").json()["sessions_active"] == 0

    text = "\n".join(record.getMessage() for record in caplog.records)
    for step in (
        "step=twilio.connected",
        "step=twilio.start",
        "step=tts.start",
        "step=tts.done",
        "step=twilio.media.in",
        "step=vad.speech_started",
        "step=vad.speech_ended",
        "step=stt.done",
        "step=llm.start",
        "step=llm.done",
        "step=twilio.stop",
        "step=session.close",
        "step=session.removed",
        "step=twilio.ws.cleanup",
    ):
        assert step in text, f"log manquant: {step}"
    assert "+33612345678" not in text
    assert "0612345678" not in text
    assert runtime.stt.calls == 1
    assert "Pour quelle date ?" in runtime.tts.texts


def test_twilio_call_e2e_disconnect_without_stop_cleans_session():
    app, _runtime = _build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/media-stream") as ws:
            ws.send_text(_start_msg())
            _receive_media(ws, GREETING_FRAMES)
            assert app.state.sessions.active_count == 1
            active = client.get(
                "/monitoring",
                headers={"x-api-key": "test-key"},
            ).json()["active_sessions"]
            assert active == [
                {
                    "streamSid": STREAM_SID,
                    "callSid": CALL_SID,
                    "instanceId": "restaurant-e2e",
                    "stage": "listening",
                    "duration_seconds": active[0]["duration_seconds"],
                    "provider": "vllm",
                    "last_error": None,
                }
            ]
            assert "telephone" not in active[0]
            assert "callerNumber" not in active[0]
        assert app.state.sessions.active_count == 0


def test_twilio_two_calls_leave_no_session():
    app, _runtime = _build_app()
    with TestClient(app) as client:
        for _ in range(2):
            with client.websocket_connect("/media-stream") as ws:
                ws.send_text(_start_msg())
                _receive_media(ws, GREETING_FRAMES)
                ws.send_text(json.dumps({"event": "stop", "streamSid": STREAM_SID}))
            assert app.state.sessions.active_count == 0


def test_twilio_e2e_tool_round_uses_fastify_dispatcher():
    llm = FakeLLM(LLMResult(content="Il reste des places a midi."), tool_then_text=True)
    app, runtime = _build_app(llm=llm)
    with TestClient(app) as client:
        with client.websocket_connect("/media-stream") as ws:
            ws.send_text(_start_msg())
            _receive_media(ws, GREETING_FRAMES)
            ws.send_text(_media_msg())
            ws.send_text(_media_msg())
            _receive_media(ws, GREETING_FRAMES)
            ws.send_text(json.dumps({"event": "stop", "streamSid": STREAM_SID}))
        assert runtime.node.availability_calls == ["2026-09-09"]
        assert app.state.sessions.active_count == 0
        assert "Il reste des places a midi." in runtime.tts.texts
        hangup = runtime.node.hangups[0]["payload"]
        assert hangup["liveCreate"] is None
        assert hangup["instanceId"] == "restaurant-e2e"
