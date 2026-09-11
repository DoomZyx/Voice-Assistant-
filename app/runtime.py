from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import LLMProvider
from app.node_client import NodeApiClient
from app.stt import STTEngine
from app.telemetry import VoiceTelemetry
from app.tools.dispatcher import ToolDispatcher
from app.tts import TTSEngine
from app.vad.silero import SileroModel


@dataclass
class VoiceRuntime:
    settings: Settings
    node: NodeApiClient
    dispatcher: ToolDispatcher
    llm: LLMProvider | None
    stt: STTEngine | None
    tts: TTSEngine | None
    vad_model: SileroModel | None
    engines_ready: bool
    engine_error: str | None = None
    telemetry: VoiceTelemetry | None = None
    component_errors: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.telemetry is None:
            self.telemetry = VoiceTelemetry()
        if self.component_errors is None:
            self.component_errors = {}

    @property
    def active_llm_provider(self) -> str:
        provider = getattr(self.llm, "active_provider", None)
        return str(provider or self.settings.llm_provider)

    def engine_states(self) -> dict[str, dict[str, Any]]:
        components = {
            "vad": self.vad_model,
            "stt": self.stt,
            "llm": self.llm,
            "tts": self.tts,
        }
        errors = self.component_errors or {}
        globally_unready = not self.engines_ready and not errors
        return {
            name: {
                "ready": engine is not None and name not in errors and not globally_unready,
                "error": errors.get(name) or (self.engine_error if engine is None else None),
            }
            for name, engine in components.items()
        }
