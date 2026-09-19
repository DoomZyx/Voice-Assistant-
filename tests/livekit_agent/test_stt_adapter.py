from types import SimpleNamespace

import numpy as np
import pytest

from livekit_agent.stt.whisper import _buffer_to_frames, transcribe_audio_frames


class FakeUtteranceSTT:
    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        self.seen.append(np.asarray(pcm16_16k, dtype=np.int16).copy())
        return "bonjour"


@pytest.mark.asyncio
async def test_whisper_adapter_transcribes_pcm16_16k_with_fake_engine():
    engine = FakeUtteranceSTT()
    pcm = np.array([1, 2, 3, 4], dtype=np.int16)
    text = await transcribe_audio_frames(engine, [(pcm.tobytes(), 16000, 1)])
    assert text == "bonjour"
    assert len(engine.seen) == 1
    assert engine.seen[0].tolist() == [1, 2, 3, 4]


def test_buffer_to_frames_reads_livekit_like_objects():
    frame = SimpleNamespace(data=b"\x01\x00\x02\x00", sample_rate=16000, num_channels=1)
    frames = _buffer_to_frames(frame)
    assert frames == [(b"\x01\x00\x02\x00", 16000, 1)]


def test_buffer_to_frames_accepts_tuple_of_frames():
    left = SimpleNamespace(data=b"\x01\x00", sample_rate=48000, num_channels=1)
    right = SimpleNamespace(data=b"\x02\x00", sample_rate=48000, num_channels=1)
    frames = _buffer_to_frames((left, right))
    assert frames == [(b"\x01\x00", 48000, 1), (b"\x02\x00", 48000, 1)]


@pytest.mark.asyncio
async def test_build_whisper_stt_is_non_streaming_when_livekit_present():
    livekit_agents = pytest.importorskip("livekit.agents")
    from livekit_agent.config import LiveKitSettings
    from livekit_agent.stt.whisper import build_whisper_stt

    stt = build_whisper_stt(LiveKitSettings(_env_file=None), engine=FakeUtteranceSTT())
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False
    assert hasattr(livekit_agents, "stt")
