from typing import Protocol

import numpy as np


class STTEngine(Protocol):
    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        """V1 : utterance complete. V2 : remplacer par un stream de partiels."""
        ...
