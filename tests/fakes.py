from __future__ import annotations

import numpy as np

from app.llm import LLMResult, ToolCall
from app.vad import VADResult


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        self.sent.append(message)


class FakeVadSession:
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
            return VADResult(True, True, False, 0.9)
        if self.n >= self.end_on:
            self.in_speech = False
            return VADResult(True, False, True, 0.1)
        return VADResult(True, False, False, 0.8)


class FakeVadModel:
    def __init__(self, end_on: int = 2) -> None:
        self.end_on = end_on

    def new_session(self, settings) -> FakeVadSession:
        return FakeVadSession(self.end_on)


class FakeSTT:
    def __init__(self, text: str = "bonjour") -> None:
        self.text = text
        self.calls = 0

    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        self.calls += 1
        return self.text


class FakeLLM:
    def __init__(self, result: LLMResult | None = None, tool_then_text: bool = False) -> None:
        self.calls = 0
        self._result = result or LLMResult(content="Oui, je vous ecoute.")
        self._tool_then_text = tool_then_text

    async def complete(self, messages, tools) -> LLMResult:
        self.calls += 1
        if self._tool_then_text and self.calls == 1:
            return LLMResult(
                tool_calls=[
                    ToolCall(id="call_1", name="check_availability", arguments={"date": "2026-09-09"})
                ]
            )
        return self._result


class FakeTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(400, dtype=np.int16)


class FakeNode:
    def __init__(self) -> None:
        self.availability_calls: list[str] = []
        self.appointments: list[dict] = []
        self.voice_contexts: dict[str, dict] = {}

    async def get_voice_context(self, instance_id: str) -> dict:
        return self.voice_contexts.get(instance_id, {})

    async def check_availability(self, date: str) -> dict:
        self.availability_calls.append(date)
        return {"success": True, "date": date, "slots": ["12:00"]}

    async def create_appointment(self, payload: dict) -> dict:
        self.appointments.append(payload)
        return {"success": True, "appointment": {"_id": "1"}}
