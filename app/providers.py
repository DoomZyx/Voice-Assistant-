from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import LLMProvider
from app.stt import STTEngine
from app.turn.detector import HeuristicTurnDetector, SilenceTurnDetector, SmartTurnDetector, TurnDetector
from app.tts import TTSEngine
from app.vad.silero import SileroModel


STT_PROVIDERS = ("faster-whisper", "whisper-large-v3-turbo", "parakeet")
TTS_PROVIDERS = ("kokoro",)
LLM_PROVIDERS = ("vllm", "openai", "openai-compat")
VAD_PROVIDERS = ("silero",)
TURN_PROVIDERS = ("silence", "heuristic", "smart_turn")
ORCHESTRATORS = ("native", "pipecat")


@dataclass
class EngineBundle:
    stt: STTEngine | None
    tts: TTSEngine | None
    llm: LLMProvider | None
    vad_model: SileroModel | None
    turn: TurnDetector
    stt_provider: str
    tts_provider: str
    llm_provider: str
    vad_provider: str
    turn_provider: str
    orchestrator: str


def available_providers() -> dict[str, tuple[str, ...]]:
    return {
        "stt": STT_PROVIDERS,
        "tts": TTS_PROVIDERS,
        "llm": LLM_PROVIDERS,
        "vad": VAD_PROVIDERS,
        "turn": TURN_PROVIDERS,
        "orchestrator": ORCHESTRATORS,
    }


def build_turn_detector(settings: Settings) -> TurnDetector:
    name = (settings.turn_detector or "heuristic").strip().lower()
    if name == "silence":
        return SilenceTurnDetector()
    if name == "smart_turn":
        return SmartTurnDetector()
    if name == "heuristic":
        return HeuristicTurnDetector()
    raise ValueError(f"TURN_DETECTOR inconnu: {settings.turn_detector}")


def resolve_whisper_model(settings: Settings) -> str:
    provider = (settings.stt_provider or "faster-whisper").strip().lower()
    if provider == "whisper-large-v3-turbo":
        return "large-v3-turbo"
    return settings.whisper_model_size


def build_engines(
    settings: Settings,
    *,
    stt: STTEngine | None = None,
    tts: TTSEngine | None = None,
    llm: LLMProvider | None = None,
    vad_model: SileroModel | None = None,
    eager: bool = False,
) -> EngineBundle:
    stt_name = (settings.stt_provider or "faster-whisper").strip().lower()
    tts_name = (settings.tts_provider or "kokoro").strip().lower()
    llm_name = (settings.llm_provider or "vllm").strip().lower()
    vad_name = (settings.vad_provider or "silero").strip().lower()
    orchestrator = (settings.voice_orchestrator or "native").strip().lower()

    if stt_name not in STT_PROVIDERS:
        raise ValueError(f"STT_PROVIDER inconnu: {settings.stt_provider}")
    if tts_name not in TTS_PROVIDERS:
        raise ValueError(f"TTS_PROVIDER inconnu: {settings.tts_provider}")
    if llm_name not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER inconnu: {settings.llm_provider}")
    if vad_name not in VAD_PROVIDERS:
        raise ValueError(f"VAD_PROVIDER inconnu: {settings.vad_provider}")
    if orchestrator not in ORCHESTRATORS:
        raise ValueError(f"VOICE_ORCHESTRATOR inconnu: {settings.voice_orchestrator}")

    turn = build_turn_detector(settings)
    if stt is not None or tts is not None or llm is not None or vad_model is not None:
        return EngineBundle(
            stt=stt,
            tts=tts,
            llm=llm,
            vad_model=vad_model,
            turn=turn,
            stt_provider=stt_name,
            tts_provider=tts_name,
            llm_provider=llm_name,
            vad_provider=vad_name,
            turn_provider=turn.name,
            orchestrator=orchestrator,
        )
    if not eager:
        return EngineBundle(
            stt=None,
            tts=None,
            llm=None,
            vad_model=None,
            turn=turn,
            stt_provider=stt_name,
            tts_provider=tts_name,
            llm_provider=llm_name,
            vad_provider=vad_name,
            turn_provider=turn.name,
            orchestrator=orchestrator,
        )
    return _load_eager_engines(settings, turn)


def _load_eager_engines(settings: Settings, turn: TurnDetector) -> EngineBundle:
    from app.stt.faster_whisper import FasterWhisperSTT
    from app.tts.kokoro import KokoroTTS
    from app.vad.silero import load_silero_model

    stt_name = (settings.stt_provider or "faster-whisper").strip().lower()
    if stt_name == "parakeet":
        from app.stt.parakeet import ParakeetSTT

        stt: Any = ParakeetSTT(settings)
    else:
        model_settings = settings.model_copy(update={"whisper_model_size": resolve_whisper_model(settings)})
        stt = FasterWhisperSTT(model_settings)
    return EngineBundle(
        stt=stt,
        tts=KokoroTTS(settings),
        llm=None,
        vad_model=load_silero_model(settings),
        turn=turn,
        stt_provider=stt_name,
        tts_provider="kokoro",
        llm_provider=(settings.llm_provider or "vllm").strip().lower(),
        vad_provider="silero",
        turn_provider=turn.name,
        orchestrator=(settings.voice_orchestrator or "native").strip().lower(),
    )
