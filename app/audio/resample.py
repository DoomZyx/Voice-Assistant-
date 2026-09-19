from __future__ import annotations

import numpy as np


def resample_pcm16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm.astype(np.int16, copy=False)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.int16)
    source = pcm.astype(np.float32)
    if dst_rate < src_rate:
        window = max(2, int(round(src_rate / dst_rate)))
        kernel = np.ones(window, dtype=np.float32) / window
        source = np.convolve(source, kernel, mode="same")
    n = max(1, int(round(pcm.size * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, num=source.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    y = np.interp(x_new, x_old, source)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)
    return np.clip(np.rint(audio * 32767.0), -32768, 32767).astype(np.int16)
