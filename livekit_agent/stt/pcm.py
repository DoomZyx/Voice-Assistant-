from __future__ import annotations

import numpy as np

from app.audio.resample import resample_pcm16


def frames_to_pcm16(frames: list[tuple[bytes, int, int]]) -> tuple[np.ndarray, int]:
    if not frames:
        return np.zeros(0, dtype=np.int16), 16000
    sample_rate = frames[0][1]
    chunks: list[np.ndarray] = []
    for data, rate, channels in frames:
        if rate != sample_rate:
            raise ValueError("sample rate mismatch in STT frames")
        pcm = np.frombuffer(data, dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        chunks.append(pcm)
    return np.concatenate(chunks), sample_rate


def resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    return resample_pcm16(pcm, sample_rate, 16000)
