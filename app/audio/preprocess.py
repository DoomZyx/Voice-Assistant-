from __future__ import annotations

import numpy as np


TTS_PEAK = 0.72


def peak_normalize_pcm16(pcm16: np.ndarray, peak: float = TTS_PEAK) -> np.ndarray:
    """Evite la saturation mu-law en baissant seulement les pics trop hauts."""
    audio = np.asarray(pcm16, dtype=np.int16)
    if audio.size == 0:
        return audio
    max_abs = int(np.max(np.abs(audio)))
    if max_abs <= 0:
        return audio
    target = int(32767 * peak)
    if max_abs <= target:
        return audio
    scaled = audio.astype(np.float32) * (target / max_abs)
    return np.clip(np.rint(scaled), -32768, 32767).astype(np.int16)


def apply_preprocess(pcm16: np.ndarray, mode: str = "off") -> np.ndarray:
    audio = np.asarray(pcm16, dtype=np.int16)
    name = (mode or "off").strip().lower()
    if name in {"", "off", "none"}:
        return audio
    if name == "agc":
        return _agc(audio)
    if name == "gate":
        return _noise_gate(audio)
    raise ValueError(f"AUDIO_PREPROCESS inconnu: {mode}")


def _agc(pcm16: np.ndarray, target: float = 0.15) -> np.ndarray:
    if pcm16.size == 0:
        return pcm16
    samples = pcm16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < 1e-4:
        return pcm16
    gain = min(8.0, target / rms)
    scaled = np.clip(samples * gain, -1.0, 1.0)
    return np.clip(np.rint(scaled * 32767.0), -32768, 32767).astype(np.int16)


def _noise_gate(pcm16: np.ndarray, threshold: float = 0.012) -> np.ndarray:
    if pcm16.size == 0:
        return pcm16
    samples = pcm16.astype(np.float32) / 32768.0
    mask = np.abs(samples) >= threshold
    gated = samples * mask.astype(np.float32)
    return np.clip(np.rint(gated * 32767.0), -32768, 32767).astype(np.int16)
