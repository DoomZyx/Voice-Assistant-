import base64
import json

import numpy as np
import pytest

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.llm import LLMResult
from app.runtime import VoiceRuntime
from app.session.call import CallSession
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeLLM, FakeSTT, FakeTTS, FakeVadModel, FakeWebSocket, FakeNode


def _runtime(llm=None, stt=None, tts=None) -> VoiceRuntime:
    settings = Settings(voice_eager_load=False, greeting_text="Bonjour, je vous ecoute.")
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
