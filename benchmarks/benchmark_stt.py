"""Benchmark ASR sur corpus telephone mu-law 8 kHz."""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from statistics import median

import numpy as np

from app.audio.preprocess import apply_preprocess
from app.audio.resample import resample_pcm16
from app.config import Settings
from benchmarks.wer import summarize_conditions, word_error_rate


ROOT = Path(__file__).resolve().parent


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        return np.frombuffer(frames, dtype=np.int16)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((quantile / 100) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


async def run_benchmark(provider: str, preprocess: str) -> dict:
    settings = Settings(
        stt_provider=provider,
        whisper_model_size="large-v3-turbo" if provider == "whisper-large-v3-turbo" else Settings().whisper_model_size,
        audio_preprocess=preprocess,
    )
    from app.providers import resolve_whisper_model

    if provider == "parakeet":
        from app.stt.parakeet import ParakeetSTT

        stt = ParakeetSTT(settings)
        model = settings.parakeet_model
    else:
        from app.stt.faster_whisper import FasterWhisperSTT

        model_settings = settings.model_copy(update={"whisper_model_size": resolve_whisper_model(settings)})
        stt = FasterWhisperSTT(model_settings)
        model = model_settings.whisper_model_size

    manifest_path = ROOT / "audio" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Corpus absent. Lancer: python -m benchmarks.generate_corpus")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    latencies: list[float] = []
    wers: list[float] = []
    details = []
    for row in rows:
        pcm8k = apply_preprocess(_read_wav(ROOT / row["path"]), preprocess)
        pcm16k = resample_pcm16(pcm8k, 8000, 16000)
        started = time.perf_counter()
        hypothesis = await stt.transcribe_utterance(pcm16k)
        elapsed = (time.perf_counter() - started) * 1000
        duration_s = pcm8k.size / 8000 if pcm8k.size else 0.001
        wer = word_error_rate(row["text"], hypothesis)
        latencies.append(elapsed)
        wers.append(wer)
        details.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "reference": row["text"],
                "hypothesis": hypothesis,
                "wer": round(wer, 4),
                "latency_ms": round(elapsed, 1),
                "rtf": round((elapsed / 1000) / duration_s, 3),
                "partial_first_ms": None,
            }
        )
    summary = {
        "provider": provider,
        "model": model,
        "preprocess": preprocess,
        "samples": len(details),
        "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
        "latency_p50_ms": round(median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(_percentile(latencies, 95), 1) if latencies else None,
        "rtf_mean": round(sum(item["rtf"] for item in details) / len(details), 3) if details else None,
        "hardware": "cpu",
        "by_condition": summarize_conditions(details),
        "details": details,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"stt_{provider}_{preprocess}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="faster-whisper")
    parser.add_argument("--preprocess", default="off")
    args = parser.parse_args()
    import asyncio

    summary = asyncio.run(run_benchmark(args.provider, args.preprocess))
    print(json.dumps({key: summary[key] for key in summary if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
