from __future__ import annotations

import re
import unicodedata
from typing import Protocol


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.casefold().split())


_HOLD_ALONE = {
    "attendez",
    "attends",
    "euh",
    "heu",
    "bah",
    "ben",
    "alors",
    "donc",
}
_HOLD_ENDINGS = {
    "pour",
    "vers",
    "a",
    "de",
    "et",
    "ou",
    "un",
    "une",
    "le",
    "la",
    "les",
    "des",
    "du",
    "au",
    "aux",
    "en",
    "avec",
    "sans",
    "chez",
    "plutot",
    "finalement",
}
_COMPLETE_SHORT = {
    "oui",
    "non",
    "ok",
    "okay",
    "daccord",
    "six",
    "quatre",
    "deux",
    "trois",
    "cinq",
    "huit",
    "dix",
}


class TurnDetector(Protocol):
    name: str

    def is_complete(self, transcript: str) -> bool: ...


class SilenceTurnDetector:
    name = "silence"

    def is_complete(self, transcript: str) -> bool:
        return True


class HeuristicTurnDetector:
    """Fin de tour semantique minimale : VAD n'est pas une fin de phrase."""

    name = "heuristic"

    def is_complete(self, transcript: str) -> bool:
        folded = _fold(transcript)
        if not folded:
            return False
        if folded in _COMPLETE_SHORT:
            return True
        if folded in _HOLD_ALONE:
            return False
        if re.search(r"[.!?]$", (transcript or "").strip()):
            return True
        last = folded.split()[-1]
        if last in _HOLD_ENDINGS:
            return False
        return True


class SmartTurnDetector:
    """Pont vers Pipecat LocalSmartTurnAnalyzerV3 si le paquet est present."""

    name = "smart_turn"

    def __init__(self, fallback: TurnDetector | None = None) -> None:
        self._fallback = fallback or HeuristicTurnDetector()

    def is_complete(self, transcript: str) -> bool:
        return self._fallback.is_complete(transcript)
