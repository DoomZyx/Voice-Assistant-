from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from app.audio.resample import resample_pcm16


def pcm16_at_rate(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return np.asarray(pcm, dtype=np.int16)
    return resample_pcm16(np.asarray(pcm, dtype=np.int16), int(source_rate), int(target_rate))


def pcm16_chunks(pcm: np.ndarray, *, sample_rate: int, frame_ms: int = 20) -> Iterator[bytes]:
    if pcm.size == 0:
        return
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    for start in range(0, pcm.size, frame_samples):
        yield np.asarray(pcm[start : start + frame_samples], dtype=np.int16).tobytes()
