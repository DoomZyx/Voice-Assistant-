"""Doubles A/B : importers app/ab/orchestrator.py. Pas de persistance."""

from __future__ import annotations

import numpy as np

from app.llm import LLMResult
from app.vad import VADResult


class ProbeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools) -> LLMResult:
        self.calls += 1
        return LLMResult(content="Oui. Pour combien de personnes ?")


class SequenceSTT:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return text


class InstantTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(400, dtype=np.int16)


class SlowTTS:
    """TTS long seulement hors greeting, pour mesurer le barge-in sans attendre l'accueil."""

    async def synthesize(self, text: str) -> np.ndarray:
        if text.startswith("Bonjour"):
            return np.zeros(400, dtype=np.int16)
        return np.zeros(16000, dtype=np.int16)


class CycleVad:
    def __init__(self, end_on: int = 2) -> None:
        self.n = 0
        self.end_on = end_on
        self.in_speech = False

    def reset(self) -> None:
        self.n = 0
        self.in_speech = False

    def process(self, pcm) -> VADResult:
        self.n += 1
        if self.n == 1:
            self.in_speech = True
            return VADResult(True, True, False, 0.94)
        if self.n >= self.end_on:
            self.in_speech = False
            return VADResult(True, False, True, 0.08)
        return VADResult(True, False, False, 0.85)


class AlwaysSpeechVad:
    def __init__(self) -> None:
        self.in_speech = False

    def reset(self) -> None:
        self.in_speech = False

    def process(self, pcm) -> VADResult:
        self.in_speech = True
        return VADResult(True, True, False, 0.96)


class QuietVad:
    def __init__(self, threshold: float = 80.0) -> None:
        self.in_speech = False
        self.threshold = threshold

    def reset(self) -> None:
        self.in_speech = False

    def process(self, pcm) -> VADResult:
        energy = float(np.max(np.abs(pcm))) if getattr(pcm, "size", 0) else 0.0
        if energy < self.threshold:
            self.in_speech = False
            return VADResult(False, False, False, 0.04)
        self.in_speech = True
        return VADResult(True, True, False, 0.9)


class CycleVadModel:
    def __init__(self, session) -> None:
        self._session = session

    def new_session(self, settings):
        return self._session


class FakeWs:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: int | None = None

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, code: int = 1000) -> None:
        self.closed = code


class TrackingNode:
    def __init__(self) -> None:
        self.availability_calls: list[str] = []
        self.appointments: list[dict] = []
        self.hangups: list[dict] = []
        self.voice_contexts: dict[str, dict] = {
            "restaurant-42": {
                "restaurantInfo": {"nom": "Chez AB"},
                "menu": {"tacos": {"nom": "Tacos", "produits": [{"nom": "Tacos poulet", "prix": 9}]}},
            }
        }

    async def get_voice_context(self, instance_id: str) -> dict:
        return self.voice_contexts.get(instance_id, {})

    async def get_voice_snapshot(self, instance_id: str) -> dict:
        return {"products": [{"nom": "Tacos poulet"}]}

    async def check_availability(self, date: str, tenant_id: str | None = None) -> dict:
        self.availability_calls.append(date)
        return {"success": True, "date": date, "available": True}

    async def create_appointment(self, payload: dict, tenant_id: str | None = None) -> dict:
        self.appointments.append(payload)
        return {"success": True, "id": "resa1", "type": "reservation"}

    async def report_call_ended(self, payload: dict, tenant_id: str | None = None) -> dict:
        self.hangups.append({"payload": payload, "tenant_id": tenant_id})
        return {"success": True}
