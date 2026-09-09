from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.audio.resample import pcm16_to_float32
from app.config import Settings

logger = logging.getLogger("mysmartfood-voice.stt")


class FasterWhisperSTT:
    def __init__(self, settings: Settings) -> None:
        from faster_whisper import WhisperModel

        self._language = settings.whisper_language
        self._lock = asyncio.Lock()
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info(
            "faster-whisper charge model=%s device=%s compute_type=%s",
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        )

    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        if pcm16_16k.size == 0:
            return ""
        audio = pcm16_to_float32(pcm16_16k)

        def _run() -> str:
            segments, _info = self._model.transcribe(
                audio,
                language=self._language,
                vad_filter=False,
                beam_size=1,
            )
            return "".join(segment.text for segment in segments).strip()

        async with self._lock:
            return await asyncio.to_thread(_run)
