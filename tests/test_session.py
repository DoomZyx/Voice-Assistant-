import base64
import json

import numpy as np
import pytest

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.llm import LLMResult, ToolCall
from app.runtime import VoiceRuntime
from app.session.call import CallSession, normalize_french_caller_number
from app.stream_token import create_stream_token
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeLLM, FakeSTT, FakeTTS, FakeVadModel, FakeWebSocket, FakeNode


STREAM_SECRET = "test-twilio-token"


def _runtime(llm=None, stt=None, tts=None) -> VoiceRuntime:
    settings = Settings(
        voice_eager_load=False,
        greeting_text="Bonjour, je vous ecoute.",
        twilio_auth_token=STREAM_SECRET,
    )
    node = FakeNode()
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=llm or FakeLLM(),
        stt=stt or FakeSTT(),
        tts=tts or FakeTTS(),
        vad_model=FakeVadModel(end_on=2),
        engines_ready=True,
    )


def _media_msg(samples: int = 2000) -> str:
    pcm = np.zeros(samples, dtype=np.int16)
    payload = base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")
    return json.dumps({"event": "media", "media": {"payload": payload}})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+33612345678", "0612345678"),
        ("+33 1 42 34 56 78", "0142345678"),
        ("0033761234567", "0761234567"),
        ("anonymous", None),
        ("+44123456789", None),
        ("0612345678", None),
    ],
)
def test_normalize_french_caller_number(raw, expected):
    assert normalize_french_caller_number(raw) == expected


@pytest.mark.asyncio
async def test_greeting_on_start():
    ws = FakeWebSocket()
    tts = FakeTTS()
    session = CallSession(ws, _runtime(tts=tts))
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1","callSid":"CA1"}}')
    await session._turn_task
    assert tts.texts[0] == "Bonjour, je vous ecoute."
    assert any('"event": "media"' in item for item in ws.sent)
    await session.close()


@pytest.mark.asyncio
async def test_start_with_instance_id_rejects_missing_stream_token():
    runtime = _runtime()
    runtime.node.voice_contexts["restaurant-42"] = {"restaurantInfo": {"nom": "Secret"}}
    session = CallSession(FakeWebSocket(), runtime)
    ended = await session.handle_raw(
        json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ1",
                    "customParameters": {"instanceId": "restaurant-42"},
                },
            }
        )
    )
    assert ended is True
    assert session.instance_id is None
    assert session.last_error == "stream_token_rejected"
    assert session._turn_task is None
    assert "Secret" not in session._history[0]["content"]
    await session.close()


@pytest.mark.asyncio
async def test_twilio_context_forces_caller_number_in_create_appointment():
    class CreateAppointmentLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    tool_calls=[
                        ToolCall(
                            id="create_1",
                            name="create_appointment",
                            arguments={
                                "name": "Alice",
                                "telephone": "0000000000",
                                "date": "2026-09-12",
                                "time": "12:00",
                                "type": "Réservation de table",
                                "nombrePersonnes": 2,
                            },
                        )
                    ]
                )
            return LLMResult(content="Votre reservation est creee.")

    runtime = _runtime(llm=CreateAppointmentLLM())
    runtime.node.voice_contexts["restaurant-42"] = {
        "restaurantInfo": {"nom": "Restaurant Test"},
        "menu": {
            "burgers": {
                "nom": "Burgers",
                "produits": [{"nom": "Burger Test", "prix": 12}],
            }
        },
        "amenities": [{"slug": "pmr", "label": "Acces PMR", "status": "available"}],
    }
    session = CallSession(FakeWebSocket(), runtime)
    start = {
        "event": "start",
        "start": {
            "streamSid": "MZ1",
            "callSid": "CA1",
            "customParameters": {
                "callerNumber": "+33612345678",
                "instanceId": "restaurant-42",
                "streamToken": create_stream_token("restaurant-42", STREAM_SECRET),
            },
        },
    }
    await session.handle_raw(json.dumps(start))
    await session._turn_task
    await session._think("Oui, je confirme.")

    assert runtime.node.appointments[0]["telephone"] == "0612345678"
    assert "0612345678" in session._history[0]["content"]
    assert "Ne demande pas son numero" in session._history[0]["content"]
    assert "Restaurant Test" in session._history[0]["content"]
    assert "Burger Test" in session._history[0]["content"]
    snapshot = session.sanitized_snapshot()
    assert snapshot["instanceId"] == "restaurant-42"
    assert "0612345678" not in json.dumps(snapshot)
    assert "pmr" in session._history[0]["content"]
    await session.close()
    hangup = runtime.node.hangups[0]["payload"]
    assert hangup["liveCreate"] == {"type": "reservation", "id": "resa1"}
    assert hangup["turns"]
    assert all(turn["role"] in {"user", "assistant"} for turn in hangup["turns"])


@pytest.mark.asyncio
async def test_masked_caller_requires_number_without_forcing_tool_argument():
    session = CallSession(FakeWebSocket(), _runtime())
    await session.handle_raw(
        json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ1",
                    "customParameters": {"callerNumber": "private"},
                },
            }
        )
    )
    await session._turn_task
    assert "Demande le numero de telephone" in session._history[0]["content"]
    await session.close()


@pytest.mark.asyncio
async def test_turn_stt_llm_tts():
    ws = FakeWebSocket()
    stt = FakeSTT("une table pour deux")
    llm = FakeLLM(LLMResult(content="Pour quelle date ?"))
    tts = FakeTTS()
    session = CallSession(ws, _runtime(llm=llm, stt=stt, tts=tts))
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task
    await session.handle_raw(_media_msg())
    await session.handle_raw(_media_msg())
    await session._turn_task
    assert stt.calls == 1
    assert llm.calls >= 1
    assert "Pour quelle date ?" in tts.texts
    await session.close()


@pytest.mark.asyncio
async def test_tool_round():
    ws = FakeWebSocket()
    llm = FakeLLM(LLMResult(content="Il reste des places a midi."), tool_then_text=True)
    tts = FakeTTS()
    runtime = _runtime(llm=llm, tts=tts)
    session = CallSession(ws, runtime)
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task
    await session.handle_raw(_media_msg())
    await session.handle_raw(_media_msg())
    await session._turn_task
    assert runtime.node.availability_calls == ["2026-09-09"]
    assert "Il reste des places a midi." in tts.texts
    await session.close()


@pytest.mark.asyncio
async def test_hangup_without_create_reports_no_live_create():
    runtime = _runtime()
    session = CallSession(FakeWebSocket(), runtime)
    await session.handle_raw(
        json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ1",
                    "callSid": "CA1",
                    "customParameters": {
                        "instanceId": "restaurant-42",
                        "streamToken": create_stream_token("restaurant-42", STREAM_SECRET),
                    },
                },
            }
        )
    )
    await session._turn_task
    await session.close()
    hangup = runtime.node.hangups[0]["payload"]
    assert hangup["liveCreate"] is None
    assert hangup["instanceId"] == "restaurant-42"
    assert hangup["turns"]


@pytest.mark.asyncio
async def test_fail_soft_when_context_missing():
    from app.prompts import CONTEXT_UNAVAILABLE_GREETING

    runtime = _runtime()
    session = CallSession(FakeWebSocket(), runtime)
    await session.handle_raw(
        json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ1",
                    "callSid": "CA1",
                    "customParameters": {
                        "instanceId": "restaurant-42",
                        "streamToken": create_stream_token("restaurant-42", STREAM_SECRET),
                    },
                },
            }
        )
    )
    await session._turn_task
    assert session._context_ready is False
    assert session.last_error == "voice_context_unavailable"
    assert CONTEXT_UNAVAILABLE_GREETING in session._history[1]["content"]
    assert "indisponibles" in session._history[0]["content"]
    await session.close()


@pytest.mark.asyncio
async def test_native_runtime_records_realtime_latency_boundaries():
    class StreamingLLM:
        async def complete_stream(self, _messages, _tools):
            yield LLMResult(content="Pour quelle date ?")

        async def complete(self, _messages, _tools):
            return LLMResult(content="Pour quelle date ?")

    runtime = _runtime(llm=StreamingLLM(), stt=FakeSTT("Je veux une table."))
    session = CallSession(FakeWebSocket(), runtime)
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task
    await session.handle_raw(_media_msg())
    await session.handle_raw(_media_msg())
    await session._turn_task

    latencies = runtime.telemetry.snapshot()["latency_ms"]
    for stage in (
        "audio_in_to_stt_start",
        "llm_ttft",
        "tts_ttfa_excluding_load",
        "end_turn_to_first_audio",
    ):
        assert latencies[stage]["count"] >= 1
        assert "p50" in latencies[stage]
        assert "p95" in latencies[stage]
    await session.close()
