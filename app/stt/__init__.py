from typing import Protocol

import numpy as np


class STTEngine(Protocol):
    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        """V1 : utterance complete.

        V2 : un moteur streaming doit exposer des partiels et une latence
        `partial_first_ms` (premier token) en plus du WER final.
        """
        ...
