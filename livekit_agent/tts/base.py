from __future__ import annotations

from typing import Protocol

import numpy as np


class NativeTTS(Protocol):
    async def synthesize_native(self, text: str) -> tuple[np.ndarray, int]: ...
