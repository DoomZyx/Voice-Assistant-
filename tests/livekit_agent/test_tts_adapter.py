import numpy as np
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.tts.pcm import pcm16_at_rate, pcm16_chunks


class FakeNativeTTS:
    async def synthesize_native(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.int16), 24000
        return np.arange(48, dtype=np.int16), 24000


@pytest.mark.asyncio
async def test_fake_kokoro_engine_emits_24khz_pcm():
    engine = FakeNativeTTS()
    pcm, rate = await engine.synthesize_native("bonjour")
    assert rate == 24000
    chunks = list(pcm16_chunks(pcm, sample_rate=rate, frame_ms=1))
    assert len(chunks) == 2
    assert np.frombuffer(chunks[0], dtype=np.int16).size == 24


def test_pcm16_at_rate_resamples_away_from_native_engine_rate():
    pcm = np.arange(2205, dtype=np.int16)
    out = pcm16_at_rate(pcm, 22050, 24000)
    assert out.dtype == np.int16
    assert out.size == 2400


def test_livekit_tts_default_rate_is_24k_not_twilio_8k():
    settings = LiveKitSettings(_env_file=None)
    assert settings.tts_sample_rate == 24000
    assert settings.livekit_kokoro_speed == 1.2


@pytest.mark.asyncio
async def test_build_kokoro_tts_is_non_streaming_when_livekit_present():
    pytest.importorskip("livekit.agents")
    from livekit_agent.tts.kokoro import build_kokoro_tts

    tts = build_kokoro_tts(LiveKitSettings(_env_file=None), engine=FakeNativeTTS())
    assert tts.capabilities.streaming is False
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
