import asyncio
import base64
import json

import numpy as np
import pytest

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.runtime import VoiceRuntime
from app.session.call import CallSession
from app.tools.dispatcher import ToolDispatcher
from app.vad import VADResult
from tests.fakes import FakeLLM, FakeSTT, FakeWebSocket, FakeNode


class BargeVadSession:
    def __init__(self) -> None:
        self.in_speech = False

    def reset(self) -> None:
        self.in_speech = False

    def process(self, pcm) -> VADResult:
        self.in_speech = True
        return VADResult(True, True, False, 0.96)


class BargeVadModel:
    def new_session(self, settings) -> BargeVadSession:
        return BargeVadSession()


class SlowTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(8000, dtype=np.int16)


def _media(samples: int = 2000, amplitude: int = 0) -> str:
    pcm = np.full(samples, amplitude, dtype=np.int16) if amplitude else np.zeros(samples, dtype=np.int16)
    payload = base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")
    return json.dumps({"event": "media", "media": {"payload": payload}})


def _runtime(tts=None, confirm_ms: int = 160) -> VoiceRuntime:
    settings = Settings(
        voice_eager_load=False,
        greeting_text="Bonjour",
        barge_in_enabled=True,
        barge_in_confirm_ms=confirm_ms,
        tts_stream=False,
    )
    node = FakeNode()
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=FakeLLM(),
        stt=FakeSTT(),
        tts=tts or SlowTTS(),
        vad_model=BargeVadModel(),
        engines_ready=True,
    )


def _has_clear(sent: list[str]) -> bool:
    return any('"event": "clear"' in item or '"event":"clear"' in item for item in sent)


def _media_after_last_clear(sent: list[str]) -> int:
    after = 0
    seen = False
    for item in sent:
        if '"event": "clear"' in item or '"event":"clear"' in item:
            seen = True
            after = 0
            continue
        if seen and ('"event": "media"' in item or '"event":"media"' in item):
            after += 1
    return after


@pytest.mark.asyncio
async def test_user_speech_during_tts_clears_twilio_and_stops_playback():
    ws = FakeWebSocket()
    session = CallSession(ws, _runtime())
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task

    speak_task = asyncio.create_task(
        session._speak("J'ai une table disponible pour quatre personnes a vingt heures.")
    )
    await asyncio.sleep(0.05)
    assert session._assistant_speaking is True
    await session.handle_raw(_media(1600, 12000))
    await asyncio.wait_for(speak_task, timeout=2)

    assert _has_clear(ws.sent)
    assert session._assistant_speaking is False
    assert _media_after_last_clear(ws.sent) == 0
    await session.close()


@pytest.mark.asyncio
async def test_50ms_burst_does_not_clear_during_tts():
    ws = FakeWebSocket()
    session = CallSession(ws, _runtime(confirm_ms=160))
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task
    speak_task = asyncio.create_task(session._speak("J'ai une table disponible a vingt heures."))
    await asyncio.sleep(0.05)
    await session.handle_raw(_media(400, 12000))
    await asyncio.wait_for(speak_task, timeout=3)
    assert _has_clear(ws.sent) is False
    await session.close()
