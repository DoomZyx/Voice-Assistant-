from __future__ import annotations

from typing import Protocol

import numpy as np


class UtteranceSTT(Protocol):
    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str: ...
