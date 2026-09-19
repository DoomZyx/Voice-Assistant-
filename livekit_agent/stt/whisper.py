from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from livekit_agent.config import LiveKitSettings
from livekit_agent.observability.audio_path import log_stt_resample
from livekit_agent.stt.base import UtteranceSTT
from livekit_agent.stt.pcm import frames_to_pcm16, resample_to_16k


def _legacy_settings(settings: LiveKitSettings):
    from app.config import Settings

    return Settings(
        whisper_model_size=settings.whisper_model_size,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        whisper_language=settings.whisper_language,
        whisper_initial_prompt=settings.whisper_initial_prompt,
        whisper_beam_size=settings.whisper_beam_size,
        cache_dir=settings.cache_dir,
    )


def build_whisper_engine(settings: LiveKitSettings) -> UtteranceSTT:
    from app.stt.faster_whisper import FasterWhisperSTT

    return FasterWhisperSTT(_legacy_settings(settings))


async def transcribe_audio_frames(engine: UtteranceSTT, frames: list[tuple[bytes, int, int]]) -> str:
    pcm, rate = frames_to_pcm16(frames)
    channels = frames[0][2] if frames else 1
    log_stt_resample(inbound_hz=rate, channels=channels, whisper_hz=16000)
    return await engine.transcribe_utterance(resample_to_16k(pcm, rate))


def _buffer_to_frames(buffer: Any) -> list[tuple[bytes, int, int]]:
    if isinstance(buffer, Sequence) and not isinstance(buffer, (bytes, bytearray, str)):
        frames = list(buffer)
    else:
        frames = [buffer]
    out: list[tuple[bytes, int, int]] = []
    for frame in frames:
        data = bytes(getattr(frame, "data", b""))
        rate = int(getattr(frame, "sample_rate", 16000))
        channels = int(getattr(frame, "num_channels", 1) or 1)
        out.append((data, rate, channels))
    return out


def build_whisper_stt(settings: LiveKitSettings, engine: UtteranceSTT | None = None):
    from livekit.agents import stt
    from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

    runtime = engine or build_whisper_engine(settings)

    class FasterWhisperLiveKitSTT(stt.STT):
        def __init__(self) -> None:
            super().__init__(
                capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
            )
            self._engine = runtime
            self._language = settings.whisper_language

        @property
        def model(self) -> str:
            return settings.whisper_model_size

        @property
        def provider(self) -> str:
            return "faster-whisper"

        async def _recognize_impl(
            self,
            buffer: Any,
            *,
            language: NotGivenOr[str] = NOT_GIVEN,
            conn_options: APIConnectOptions,
        ) -> stt.SpeechEvent:
            chosen = language if language != NOT_GIVEN and language else self._language
            text = await transcribe_audio_frames(self._engine, _buffer_to_frames(buffer))
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text=text, language=str(chosen or "fr"))],
            )

    return FasterWhisperLiveKitSTT()
