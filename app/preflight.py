"""Controle brique par brique. A lancer avant tout appel Twilio."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings, apply_huggingface_token, get_settings

logger = logging.getLogger("mysmartfood-voice.preflight")


@dataclass
class BrickResult:
    name: str
    ok: bool
    ms: int
    detail: str


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def check_fastapi(settings: Settings) -> BrickResult:
    started = time.perf_counter()
    try:
        from app.main import create_app

        light = settings.model_copy(update={"voice_eager_load": False})
        app = create_app(light)
        with TestClient(app) as client:
            response = client.get("/health")
        body = response.json()
        if response.status_code != 200:
            return BrickResult("fastapi", False, _ms(started), f"HTTP {response.status_code}")
        return BrickResult(
            "fastapi",
            True,
            _ms(started),
            f"health={body.get('status')} sessions={body.get('sessions_active')}",
        )
    except Exception as exc:
        return BrickResult("fastapi", False, _ms(started), str(exc))


async def check_silero(settings: Settings) -> BrickResult:
    started = time.perf_counter()
    try:
        from app.vad.silero import WINDOW_8K, load_silero_model

        model = load_silero_model(settings)
        session = model.new_session(settings)
        pcm = np.zeros(WINDOW_8K * 4, dtype=np.int16)
        result = session.process(pcm)
        return BrickResult(
            "silero",
            True,
            _ms(started),
            f"onnx charge probability={result.probability:.3f}",
        )
    except Exception as exc:
        return BrickResult("silero", False, _ms(started), str(exc))


async def check_whisper(settings: Settings) -> BrickResult:
    started = time.perf_counter()
    if importlib.util.find_spec("faster_whisper") is None:
        return BrickResult(
            "whisper",
            False,
            _ms(started),
            "faster-whisper absent. pip install -e '.[voice]'",
        )
    try:
        from app.stt.faster_whisper import FasterWhisperSTT

        stt = FasterWhisperSTT(settings)
        audio = np.zeros(settings.stt_sample_rate, dtype=np.int16)
        text = await stt.transcribe_utterance(audio)
        return BrickResult(
            "whisper",
            True,
            _ms(started),
            (
                f"device={settings.whisper_device} compute_type={settings.whisper_compute_type} "
                f"model={settings.whisper_model_size} text={text!r}. "
                "CPU/int8 = test fonctionnel. GPU: WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16"
            ),
        )
    except Exception as exc:
        return BrickResult("whisper", False, _ms(started), str(exc))


async def check_vllm(settings: Settings) -> BrickResult:
    started = time.perf_counter()
    from app.llm.vllm import VLLMProvider, model_id_error

    llm = VLLMProvider(settings)
    try:
        served = await llm.list_models()
        mismatch = model_id_error(settings.vllm_model, served)
        if mismatch:
            return BrickResult("vllm", False, _ms(started), mismatch)
        result = await llm.complete(
            [{"role": "user", "content": "Reponds exactement: ok"}],
            tools=[],
        )
        return BrickResult(
            "vllm",
            True,
            _ms(started),
            f"model={settings.vllm_model} served={served} reply={result.content!r}"[:400],
        )
    except Exception as exc:
        return BrickResult(
            "vllm",
            False,
            _ms(started),
            (
                f"{exc}. Verifier que vLLM tourne sur {settings.vllm_base_url} "
                "et que VLLM_MODEL est l'id exact de GET /v1/models. "
                "Un depot Llama gated exige HF_TOKEN pour le telechargement, pas pour l'inference locale."
            ),
        )
    finally:
        await llm.aclose()


async def check_kokoro(settings: Settings) -> BrickResult:
    started = time.perf_counter()
    if importlib.util.find_spec("kokoro_onnx") is None:
        return BrickResult(
            "kokoro",
            False,
            _ms(started),
            "kokoro-onnx absent. pip install -e '.[voice]'",
        )
    try:
        from app.tts.kokoro import KokoroTTS

        tts = KokoroTTS(settings)
        pcm = await tts.synthesize("Bonjour.")
        if pcm.size == 0:
            return BrickResult("kokoro", False, _ms(started), "synthese vide")
        return BrickResult(
            "kokoro",
            True,
            _ms(started),
            f"voice={settings.kokoro_voice} samples={pcm.size} rate={settings.twilio_sample_rate}",
        )
    except Exception as exc:
        return BrickResult("kokoro", False, _ms(started), str(exc))


async def run_preflight(settings: Settings | None = None) -> list[BrickResult]:
    settings = settings or get_settings()
    apply_huggingface_token(settings)
    checks = (
        check_fastapi,
        check_silero,
        check_whisper,
        check_vllm,
        check_kokoro,
    )
    results: list[BrickResult] = []
    for check in checks:
        result = await check(settings)
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        logger.info("preflight brick=%s status=%s ms=%s detail=%s", result.name, status, result.ms, result.detail)
        print(f"[{result.name:8}] {status:4} {result.ms}ms  {result.detail}")
    return results


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    results = asyncio.run(run_preflight())
    failed = [item.name for item in results if not item.ok]
    print()
    if failed:
        print(f"Preflight FAIL: {', '.join(failed)}. Ne pas lancer d'appel Twilio tant que ces briques echouent.")
        raise SystemExit(1)
    print("Preflight OK. Les briques locales repondent. Tu peux tester un appel Twilio.")


if __name__ == "__main__":
    run()
