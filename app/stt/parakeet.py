from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.audio.resample import pcm16_to_float32
from app.config import Settings


logger = logging.getLogger("mysmartfood-voice.stt")


class ParakeetSTT:
    """Candidat NVIDIA Parakeet. Charge NeMo seulement si le paquet est installe."""

    def __init__(self, settings: Settings) -> None:
        self._language = settings.whisper_language
        self._lock = asyncio.Lock()
        self._model = None
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError(
                "STT_PROVIDER=parakeet exige nemo_toolkit[asr]. "
                "Garder faster-whisper tant que le benchmark Parakeet n'est pas mesure."
            ) from exc
        model_id = settings.parakeet_model or "nvidia/parakeet-tdt-0.6b-v3"
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        logger.info("parakeet charge model=%s", model_id)

    async def transcribe_utterance(self, pcm16_16k: np.ndarray) -> str:
        if pcm16_16k.size == 0 or self._model is None:
            return ""
        audio = pcm16_to_float32(pcm16_16k)

        def _run() -> str:
            hypotheses = self._model.transcribe([audio])
            if not hypotheses:
                return ""
            first = hypotheses[0]
            text = getattr(first, "text", first)
            return str(text or "").strip()

        async with self._lock:
            return await asyncio.to_thread(_run)
