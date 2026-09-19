from types import SimpleNamespace

import pytest

from app.config import Settings
from app.pipeline.pipecat_runner import (
    PIPECAT_AUDIO_IN_HZ,
    PIPECAT_TWILIO_HZ,
    PIPECAT_VAD_STOP_SECS,
    _register_fastify_tools,
    _report_hangup,
    extract_twilio_ids,
    is_pipecat_available,
    pipecat_import_error,
    pipecat_turn_settings,
    run_pipecat_session,
    should_use_pipecat,
)
from app.stream_token import create_stream_token
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeNode


def test_pipecat_availability_is_explicit():
    available = is_pipecat_available()
    if available:
        assert pipecat_import_error() is None
    else:
        assert pipecat_import_error()


def test_should_use_pipecat_only_when_requested_and_installed():
    settings = Settings(voice_orchestrator="pipecat")
    assert should_use_pipecat(settings, available=True) is True
    assert should_use_pipecat(settings, available=False) is False
    assert should_use_pipecat(Settings(voice_orchestrator="native"), available=True) is False


def test_extract_twilio_ids_from_start_event():
    start = {
        "event": "start",
        "start": {
            "streamSid": "MZ123",
            "callSid": "CA456",
            "customParameters": {"instanceId": "inst_resto"},
        },
    }
    assert extract_twilio_ids(start) == ("MZ123", "CA456", "inst_resto")


@pytest.mark.asyncio
async def test_pipecat_rejects_invalid_stream_token_without_starting_pipeline():
    class ClosingSocket:
        def __init__(self) -> None:
            self.closed = None

        async def close(self, code=1000):
            self.closed = code

    settings = Settings(voice_eager_load=False, twilio_auth_token="secret")
    node = FakeNode()
    node.voice_contexts["restaurant-42"] = {"restaurantInfo": {"nom": "Secret"}}
    runtime = SimpleNamespace(settings=settings, node=node, telemetry=None, dispatcher=None)
    websocket = ClosingSocket()
    await run_pipecat_session(
        websocket,
        runtime,
        {
            "event": "start",
            "start": {
                "streamSid": "MZ1",
                "callSid": "CA1",
                "customParameters": {"instanceId": "restaurant-42"},
            },
        },
    )
    assert websocket.closed == 1008
    assert node.hangups == []


@pytest.mark.asyncio
async def test_pipecat_tools_force_caller_and_hit_fastify_dispatcher():
    node = FakeNode()
    registered: dict = {}

    class FakeLLM:
        def register_function(self, name, fn):
            registered[name] = fn

    runtime = SimpleNamespace(dispatcher=ToolDispatcher(node))
    await _register_fastify_tools(FakeLLM(), runtime, "restaurant-42", "0612345678")
    captured = {}

    async def capture(result):
        captured["result"] = result

    await registered["create_appointment"](
        {
            "arguments": {
                "name": "Alice",
                "telephone": "0000000000",
                "date": "2026-09-12",
                "time": "20:00",
                "type": "Réservation de table",
                "nombrePersonnes": 2,
            },
            "result_callback": capture,
        }
    )
    assert node.appointments[0]["telephone"] == "0612345678"
    assert captured["result"]["success"] is True


def test_pipecat_smart_turn_uses_16k_and_vad_stop_0_2():
    settings = pipecat_turn_settings()
    assert settings["twilio_hz"] == PIPECAT_TWILIO_HZ == 8000
    assert settings["audio_in_hz"] == PIPECAT_AUDIO_IN_HZ == 16000
    assert settings["vad_stop_secs"] == PIPECAT_VAD_STOP_SECS == 0.2


@pytest.mark.asyncio
async def test_pipecat_ulaw_8k_resamples_to_16k_pcm():
    if not is_pipecat_available():
        pytest.skip("pipecat-ai absent")
    import numpy as np
    from pipecat.audio.utils import create_stream_resampler, ulaw_to_pcm

    from app.audio.codec import pcm16_to_mulaw

    # SOXR stream: une trame 20 ms peut sortir vide. 200 ms produit du PCM 16 kHz.
    pcm8k = np.full(1600, 8000, dtype=np.int16)
    out = await ulaw_to_pcm(pcm16_to_mulaw(pcm8k), 8000, 16000, create_stream_resampler())
    assert out is not None
    assert len(out) >= 2000
    assert (len(out) % 2) == 0


@pytest.mark.asyncio
async def test_pipecat_hangup_is_idempotent():
    node = FakeNode()
    runtime = SimpleNamespace(node=node)
    plan = SimpleNamespace(instance_id="restaurant-42", stream_sid="MZ1", call_sid="CA1")
    await _report_hangup(runtime, plan)
    await _report_hangup(runtime, plan)
    assert len(node.hangups) == 1
