"""Test opt-in du flux Twilio simulé contre un serveur vLLM local réel."""

from __future__ import annotations

import base64
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.llm.vllm import VLLMProvider
from app.main import create_app
from app.runtime import VoiceRuntime
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeNode, FakeSTT, FakeTTS, FakeVadModel

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_VLLM_TESTS") != "1",
    reason="Test local vLLM désactivé",
)

STREAM_SID = "MZ_LOCAL_VLLM"
CALL_SID = "CA_LOCAL_VLLM"
GREETING_FRAME_COUNT = 3


def _start_message() -> str:
    return json.dumps(
        {
            "event": "start",
            "start": {
                "streamSid": STREAM_SID,
                "callSid": CALL_SID,
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
                "customParameters": {"instanceId": "restaurant-local"},
            },
        }
    )


def _media_message() -> str:
    pcm = np.zeros(2000, dtype=np.int16)
    payload = base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")
    return json.dumps(
        {
            "event": "media",
            "streamSid": STREAM_SID,
            "media": {"payload": payload},
        }
    )


def test_fake_twilio_websocket_uses_real_local_vllm():
    settings = Settings(
        voice_eager_load=False,
        llm_provider="vllm",
        llm_fallback_gpt=False,
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_model=os.getenv("VLLM_MODEL", "restaurant-assistant"),
        vllm_api_key=os.getenv("VLLM_API_KEY", ""),
        llm_temperature=0,
        llm_max_tokens=80,
    )
    node = FakeNode()
    llm = VLLMProvider(settings)
    tts = FakeTTS()
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=llm,
        stt=FakeSTT("Bonjour, réponds en une phrase courte sans utiliser d'outil."),
        tts=tts,
        vad_model=FakeVadModel(end_on=2),
        engines_ready=True,
    )

    with TestClient(create_app(settings, runtime)) as client:
        try:
            with client.websocket_connect("/media-stream") as websocket:
                websocket.send_text(_start_message())
                for _ in range(GREETING_FRAME_COUNT):
                    assert websocket.receive_json()["event"] == "media"

                websocket.send_text(_media_message())
                websocket.send_text(_media_message())

                response = websocket.receive_json()
                assert response["event"] == "media"
                assert response["streamSid"] == STREAM_SID
                assert response["media"]["payload"]

                websocket.send_text(
                    json.dumps({"event": "stop", "streamSid": STREAM_SID})
                )
        finally:
            client.portal.call(llm.aclose)

    assert len(tts.texts) >= 2
    assert tts.texts[1].strip()
