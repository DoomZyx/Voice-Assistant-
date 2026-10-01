import numpy as np

from app.audio.codec import mulaw_to_pcm16, pcm16_to_mulaw, split_mulaw_frames
from app.audio.resample import resample_pcm16


def test_mulaw_silence_roundtrip():
    encoded = pcm16_to_mulaw(np.zeros(8, dtype=np.int16))
    decoded = mulaw_to_pcm16(encoded)
    assert decoded.shape == (8,)
    assert int(np.max(np.abs(decoded))) < 50


def test_mulaw_non_zero_survives():
    pcm = np.array([0, 1000, -1000, 8000, -8000], dtype=np.int16)
    out = mulaw_to_pcm16(pcm16_to_mulaw(pcm))
    assert out[0] == 0
    assert np.sign(out[1]) == 1
    assert np.sign(out[2]) == -1


def test_split_mulaw_pads_last_frame():
    frames = split_mulaw_frames(b"\x00" * 200)
    assert len(frames) == 2
    assert len(frames[0]) == 160
    assert len(frames[1]) == 160
    assert frames[1][40:] == b"\xff" * 120


def test_resample_length():
    pcm = np.arange(8000, dtype=np.int16)
    out = resample_pcm16(pcm, 8000, 16000)
    assert 15900 < out.size < 16100


def _tone(freq_hz: float, rate: int, seconds: float = 0.25) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq_hz * t) * 10000).astype(np.int16)


def test_downsample_24k_to_8k_rejects_energy_above_telephone_band():
    voice = resample_pcm16(_tone(1000, 24000), 24000, 8000)
    alias = resample_pcm16(_tone(7000, 24000), 24000, 8000)
    voice_rms = float(np.sqrt(np.mean(voice.astype(np.float32) ** 2)))
    alias_rms = float(np.sqrt(np.mean(alias.astype(np.float32) ** 2)))
    assert voice_rms > 1000
    assert alias_rms < 0.15 * voice_rms
