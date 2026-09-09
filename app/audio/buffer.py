from __future__ import annotations

import numpy as np


class AudioBuffer:
    """Buffer PCM16 par session. Flush atomique pour isoler les tours de parole."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def append(self, pcm: np.ndarray) -> None:
        if pcm.size:
            self._chunks.append(np.asarray(pcm, dtype=np.int16))

    def extend_silence(self, samples: int) -> None:
        if samples > 0:
            self._chunks.append(np.zeros(samples, dtype=np.int16))

    def __len__(self) -> int:
        return int(sum(chunk.size for chunk in self._chunks))

    def flush(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        data = np.concatenate(self._chunks)
        self._chunks.clear()
        return data

    def clear(self) -> None:
        self._chunks.clear()
