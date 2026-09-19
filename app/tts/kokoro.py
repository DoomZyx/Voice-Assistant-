from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from app.audio.preprocess import peak_normalize_pcm16
from app.audio.resample import float32_to_pcm16, resample_pcm16
from app.config import Settings

logger = logging.getLogger("mysmartfood-voice.tts")

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


class KokoroTTS:
    def __init__(self, settings: Settings) -> None:
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
        logger.info("Kokoro TTS charge voice=%s", settings.kokoro_voice)

    async def synthesize(self, text: str) -> np.ndarray:
        cleaned = (text or "").strip()
        if not cleaned:
            return np.zeros(0, dtype=np.int16)

        def _run() -> np.ndarray:
            samples, sample_rate = self._kokoro.create(
                cleaned,
                voice=self._settings.kokoro_voice,
                speed=self._settings.kokoro_speed,
                lang=self._settings.kokoro_lang,
            )
            pcm = float32_to_pcm16(np.asarray(samples, dtype=np.float32))
            pcm8k = resample_pcm16(pcm, int(sample_rate), self._settings.twilio_sample_rate)
            return peak_normalize_pcm16(pcm8k)

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def synthesize_stream(self, text: str):
        pcm = await self.synthesize(text)
        if pcm.size:
            yield pcm


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Telechargement modele TTS: %s", url)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
