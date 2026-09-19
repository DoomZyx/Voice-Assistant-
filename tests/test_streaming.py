import base64
import json

import numpy as np
import pytest

from app.audio.codec import pcm16_to_mulaw
from app.config import Settings
from app.llm import LLMResult
from app.llm.vllm import iter_chat_completion_chunks
from app.runtime import VoiceRuntime
from app.session.call import CallSession
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeSTT, FakeTTS, FakeVadModel, FakeWebSocket, FakeNode


def test_iter_chat_completion_chunks_reads_sse():
    lines = [
        'data: {"choices":[{"delta":{"content":"Oui. "}}]}',
        'data: {"choices":[{"delta":{"content":"Pour combien de personnes ?"}}]}',
        "data: [DONE]",
    ]
    chunks = list(iter_chat_completion_chunks(lines))
    assert "".join(chunk.content for chunk in chunks) == "Oui. Pour combien de personnes ?"


class StreamingLLM:
    async def complete(self, messages, tools):
        return LLMResult(content="Oui. Pour combien de personnes ?")

    async def complete_stream(self, messages, tools):
        yield LLMResult(content="Oui. ")
        yield LLMResult(content="Pour combien de personnes ?")


class StreamingTTS(FakeTTS):
    async def synthesize_stream(self, text: str):
        self.texts.append(text)
        yield np.zeros(160, dtype=np.int16)


def _runtime(tts=None) -> VoiceRuntime:
    settings = Settings(voice_eager_load=False, llm_stream=True, tts_stream=True)
    node = FakeNode()
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=StreamingLLM(),
        stt=FakeSTT("bonjour"),
        tts=tts or StreamingTTS(),
        vad_model=FakeVadModel(end_on=2),
        engines_ready=True,
    )


def _media() -> str:
    payload = base64.b64encode(pcm16_to_mulaw(np.zeros(2000, dtype=np.int16))).decode("ascii")
    return json.dumps({"event": "media", "media": {"payload": payload}})


@pytest.mark.asyncio
async def test_turn_speaks_sentence_chunks_before_full_reply():
    tts = StreamingTTS()
    session = CallSession(FakeWebSocket(), _runtime(tts=tts))
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1"}}')
    await session._turn_task
    await session.handle_raw(_media())
    await session.handle_raw(_media())
    await session._turn_task
    assert tts.texts[-2:] == ["Oui.", "Pour combien de personnes ?"]
    await session.close()
