import numpy as np

from app.audio.codec import mulaw_to_pcm16, pcm16_to_mulaw
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


def test_resample_length():
    pcm = np.arange(8000, dtype=np.int16)
    out = resample_pcm16(pcm, 8000, 16000)
    assert 15900 < out.size < 16100
