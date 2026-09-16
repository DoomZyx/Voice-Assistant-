"""G.711 mu-law (Twilio Media Streams) <-> PCM16."""

from __future__ import annotations

import base64

import numpy as np

_BIAS = 0x84
_CLIP = 32635
_MULAW_FRAME_BYTES_20MS_8K = 160
MULAW_SILENCE_BYTE = 0xFF
OUTBOUND_FRAME_SECONDS = 0.02

_EXP_LUT = np.array(
    [0, 132, 396, 924, 1980, 4092, 8316, 16764],
    dtype=np.int32,
)


def mulaw_to_pcm16(mulaw: bytes | bytearray | memoryview) -> np.ndarray:
    u = np.frombuffer(bytes(mulaw), dtype=np.uint8)
    if u.size == 0:
        return np.zeros(0, dtype=np.int16)
    t = np.bitwise_not(u)
    sign = t & 0x80
    exponent = (t >> 4) & 0x07
    mantissa = t & 0x0F
    sample = _EXP_LUT[exponent] + (mantissa.astype(np.int32) << (exponent + 3))
    sample = np.where(sign != 0, -sample, sample)
    return sample.astype(np.int16)


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    if pcm.size == 0:
        return b""
    sample = pcm.astype(np.int32)
    sign = np.where(sample < 0, 0x80, 0)
    mag = np.abs(sample)
    mag = np.minimum(mag, _CLIP) + _BIAS
    exponent = np.zeros(mag.shape, dtype=np.int32)
    for exp in range(7, 0, -1):
        exponent = np.where(mag >= (0x100 << exp), exp, exponent)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    ulaw = np.bitwise_not((sign | (exponent << 4) | mantissa).astype(np.uint8))
    return ulaw.tobytes()


def mulaw_b64_to_pcm16(payload: str) -> np.ndarray:
    return mulaw_to_pcm16(base64.b64decode(payload))


def pcm16_to_mulaw_b64(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")


def split_mulaw_frames(mulaw: bytes, frame_bytes: int = _MULAW_FRAME_BYTES_20MS_8K) -> list[bytes]:
    if not mulaw:
        return []
    padded = mulaw
    remainder = len(padded) % frame_bytes
    if remainder:
        padded = padded + bytes([MULAW_SILENCE_BYTE]) * (frame_bytes - remainder)
    return [padded[i : i + frame_bytes] for i in range(0, len(padded), frame_bytes)]
