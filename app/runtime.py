from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import LLMProvider
from app.node_client import NodeApiClient
from app.stt import STTEngine
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
