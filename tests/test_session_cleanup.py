import numpy as np
import pytest

from app.config import Settings
from app.runtime import VoiceRuntime
from app.session.call import CallSession
from app.session.manager import SessionManager
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeLLM, FakeNode, FakeSTT, FakeTTS, FakeVadModel, FakeWebSocket


def _runtime() -> VoiceRuntime:
    settings = Settings(voice_eager_load=False)
    node = FakeNode()
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=FakeLLM(),
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad_model=FakeVadModel(),
        engines_ready=True,
    )


@pytest.mark.asyncio
async def test_close_clears_buffers_history_and_websocket():
    session = CallSession(FakeWebSocket(), _runtime())
    session._utterance.append(np.zeros(100, dtype=np.int16))
    session._pre_roll.append(np.zeros(50, dtype=np.int16))
    await session.close(reason="test")
    assert session.closed is True
    assert len(session._utterance) == 0
    assert len(session._pre_roll) == 0
    assert session._history == []
    assert session._ws is None
    await session.close(reason="again")
    assert session.closed is True


@pytest.mark.asyncio
async def test_manager_close_is_guaranteed_after_stop_and_disconnect():
    runtime = _runtime()
    manager = SessionManager(runtime)
    ws = FakeWebSocket()
    session = manager.create(ws)
    assert manager.active_count == 1
    await session.handle_raw('{"event":"start","start":{"streamSid":"MZ1","callSid":"CA1"}}')
    ended = await session.handle_raw('{"event":"stop","streamSid":"MZ1"}')
    assert ended is True
    assert session.closed is False
    await manager.close(ws, reason="twilio.stop")
    assert manager.active_count == 0
    assert session.closed is True
    await manager.close(ws, reason="ws.disconnect")
    assert manager.active_count == 0


@pytest.mark.asyncio
async def test_close_all_on_shutdown():
    runtime = _runtime()
    manager = SessionManager(runtime)
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    manager.create(ws1)
    manager.create(ws2)
    await manager.close_all()
    assert manager.active_count == 0
