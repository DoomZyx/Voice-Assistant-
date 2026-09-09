from typing import Protocol

import numpy as np


class TTSEngine(Protocol):
    async def synthesize(self, text: str) -> np.ndarray:
        """PCM16 8 kHz pour Twilio. V1 utterance complete. V2 : stream de frames."""
        ...
