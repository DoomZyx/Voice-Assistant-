from dataclasses import dataclass
from typing import Protocol


@dataclass
class VADResult:
    is_speech: bool
    speech_started: bool
    speech_ended: bool
    probability: float


class VADEngine(Protocol):
    def reset(self) -> None: ...

    def process(self, pcm16_8k: object) -> VADResult: ...
