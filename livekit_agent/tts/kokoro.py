from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from app.audio.preprocess import peak_normalize_pcm16
from app.audio.resample import float32_to_pcm16
from livekit_agent.config import LiveKitSettings
from livekit_agent.tts.base import NativeTTS
from livekit_agent.observability.audio_path import log_tts_resample
from livekit_agent.tts.pcm import pcm16_at_rate, pcm16_chunks

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)


class KokoroNativeTTS:
    def __init__(self, settings: LiveKitSettings) -> None:
        from kokoro_onnx import Kokoro

        cache = settings.resolve_cache_dir()
        model_path = Path(settings.kokoro_model_path) if settings.kokoro_model_path else cache / "kokoro-v1.0.onnx"
        voices_path = (
            Path(settings.kokoro_voices_path) if settings.kokoro_voices_path else cache / "voices-v1.0.bin"
        )
        if not model_path.exists():
            _download(KOKORO_MODEL_URL, model_path)
        if not voices_path.exists():
            _download(KOKORO_VOICES_URL, voices_path)
        self._settings = settings
        self._lock = asyncio.Lock()
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    async def synthesize_native(self, text: str) -> tuple[np.ndarray, int]:
        cleaned = (text or "").strip()
        if not cleaned:
            return np.zeros(0, dtype=np.int16), self._settings.tts_sample_rate

        def _run() -> tuple[np.ndarray, int]:
            samples, sample_rate = self._kokoro.create(
                cleaned,
                voice=self._settings.kokoro_voice,
                speed=self._settings.livekit_kokoro_speed,
                lang=self._settings.kokoro_lang,
            )
            pcm = peak_normalize_pcm16(float32_to_pcm16(np.asarray(samples, dtype=np.float32)))
            return pcm, int(sample_rate or self._settings.tts_sample_rate)

        async with self._lock:
            return await asyncio.to_thread(_run)


def build_kokoro_engine(settings: LiveKitSettings) -> NativeTTS:
    return KokoroNativeTTS(settings)


def build_kokoro_tts(settings: LiveKitSettings, engine: NativeTTS | None = None):
    from livekit.agents import tts, utils
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

    runtime = engine or build_kokoro_engine(settings)
    sample_rate = settings.tts_sample_rate

    class KokoroLiveKitTTS(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=sample_rate,
                num_channels=1,
            )
            self._engine = runtime

        @property
        def model(self) -> str:
            return "kokoro"

        @property
        def provider(self) -> str:
            return "kokoro-onnx"

        def synthesize(
            self,
            text: str,
            *,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        ) -> tts.ChunkedStream:
            return _KokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    class _KokoroChunkedStream(tts.ChunkedStream):
        async def _run(self, output_emitter: tts.AudioEmitter) -> None:
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            pcm, rate = await runtime.synthesize_native(self._input_text)
            used_rate = int(rate or sample_rate)
            log_tts_resample(kokoro_hz=used_rate, emit_hz=sample_rate)
            pcm = pcm16_at_rate(pcm, used_rate, sample_rate)
            for chunk in pcm16_chunks(pcm, sample_rate=sample_rate):
                output_emitter.push(chunk)
            output_emitter.flush()

    return KokoroLiveKitTTS()
