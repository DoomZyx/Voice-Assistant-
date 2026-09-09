"""Silero VAD (ONNX). Une instance de modele partagee, un etat RNN par session."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.audio.resample import pcm16_to_float32
from app.config import Settings
from app.vad import VADResult

logger = logging.getLogger("mysmartfood-voice.vad")

SILERO_ONNX_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)

WINDOW_8K = 256
CONTEXT_8K = 32


class SileroSession:
    """Etat VAD isole par appel (RNN + hysteresis parole/silence)."""

    def __init__(self, model: "SileroModel", settings: Settings) -> None:
        self._model = model
        self._threshold = settings.vad_threshold
        self._silence_samples = int(settings.twilio_sample_rate * settings.vad_silence_ms / 1000)
        self._min_speech_samples = int(
            settings.twilio_sample_rate * settings.vad_min_speech_ms / 1000
        )
        self._pending = np.zeros(0, dtype=np.int16)
        self._speech_len = 0
        self._silence_len = 0
        self._in_speech = False
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._model.reset_state(self)
        self._pending = np.zeros(0, dtype=np.int16)
        self._speech_len = 0
        self._silence_len = 0
        self._in_speech = False

    def process(self, pcm16_8k: np.ndarray) -> VADResult:
        if pcm16_8k.size == 0:
            return VADResult(False, False, False, 0.0)
        self._pending = np.concatenate([self._pending, np.asarray(pcm16_8k, dtype=np.int16)])
        speech_started = False
        speech_ended = False
        last_prob = 0.0
        is_speech_chunk = False

        while self._pending.size >= WINDOW_8K:
            window = self._pending[:WINDOW_8K]
            self._pending = self._pending[WINDOW_8K:]
            last_prob = self._model.infer(self, window)
            voiced = last_prob >= self._threshold
            is_speech_chunk = is_speech_chunk or voiced

            if voiced:
                self._speech_len += WINDOW_8K
                self._silence_len = 0
                if not self._in_speech and self._speech_len >= self._min_speech_samples:
                    self._in_speech = True
                    speech_started = True
            else:
                self._silence_len += WINDOW_8K
                if not self._in_speech:
                    self._speech_len = 0
                elif self._silence_len >= self._silence_samples:
                    self._in_speech = False
                    self._speech_len = 0
                    self._silence_len = 0
                    speech_ended = True

        return VADResult(
            is_speech=self._in_speech or is_speech_chunk,
            speech_started=speech_started,
            speech_ended=speech_ended,
            probability=last_prob,
        )


class SileroModel:
    def __init__(self, session) -> None:
        self._sess = session

    def new_session(self, settings: Settings) -> SileroSession:
        return SileroSession(self, settings)

    def reset_state(self, owner: SileroSession) -> None:
        owner._context = np.zeros((1, CONTEXT_8K), dtype=np.float32)
        owner._rnn_state = np.zeros((2, 1, 128), dtype=np.float32)
        owner._sr = np.array(8000, dtype=np.int64)
        owner._input_buffer = np.zeros((1, CONTEXT_8K + WINDOW_8K), dtype=np.float32)

    def infer(self, owner: SileroSession, window_pcm16: np.ndarray) -> float:
        audio = pcm16_to_float32(window_pcm16).reshape(1, WINDOW_8K)
        owner._input_buffer[:, :CONTEXT_8K] = owner._context
        owner._input_buffer[:, CONTEXT_8K:] = audio
        out, owner._rnn_state = self._sess.run(
            None,
            {
                "input": owner._input_buffer,
                "state": owner._rnn_state,
                "sr": owner._sr,
            },
        )
        owner._context = owner._input_buffer[:, -CONTEXT_8K:]
        return float(out.item())


def load_silero_model(settings: Settings) -> SileroModel:
    import onnxruntime as ort

    path = Path(settings.vad_model_path) if settings.vad_model_path else (
        settings.resolve_cache_dir() / "silero_vad.onnx"
    )
    if not path.exists():
        _download(SILERO_ONNX_URL, path)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)
    logger.info("Silero VAD charge: %s", path)
    return SileroModel(session)


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Telechargement modele VAD: %s", url)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
