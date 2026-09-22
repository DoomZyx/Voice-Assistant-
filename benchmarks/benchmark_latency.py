"""Mesure locale des etapes audio/VAD/TTS sans inventer de chiffres GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from app.config import Settings
from app.vad.silero import WINDOW_8K, load_silero_model


ROOT = Path(__file__).resolve().parent


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def run_latency() -> dict:
    settings = Settings()
    results: dict[str, object] = {
        "hardware": "cpu",
        "gpu": None,
        "notes": "Mesures locales CPU uniquement. vLLM et GPU L4 non mesures ici.",
    }
    started = time.perf_counter()
    vad = load_silero_model(settings)
    session = vad.new_session(settings)
    pcm = np.zeros(WINDOW_8K * 20, dtype=np.int16)
    session.process(pcm)
    results["vad_load_and_infer_ms"] = _ms(started)

    try:
        from app.tts.kokoro import KokoroTTS

        tts_started = time.perf_counter()
        tts = KokoroTTS(settings)
        pcm8k = await tts.synthesize("Oui. Pour combien de personnes ?")
        results["tts_ttfa_proxy_ms"] = _ms(tts_started)
        results["tts_samples"] = int(pcm8k.size)
    except Exception as exc:
        results["tts_error"] = str(exc)

    out = ROOT / "results" / "latency_local.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    import asyncio

    print(json.dumps(asyncio.run(run_latency()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
