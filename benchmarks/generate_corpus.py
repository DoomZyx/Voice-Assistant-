"""Synthetise le corpus restaurant puis le passe en mu-law 8 kHz."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from app.audio.codec import mulaw_to_pcm16, pcm16_to_mulaw
from app.config import Settings


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"


def _write_wav(path: Path, pcm16: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm16.tobytes())


def _mix_noise(pcm: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        return pcm
    noise = rng.normal(0.0, 400.0 if kind == "street" else 700.0, size=pcm.size)
    mixed = pcm.astype(np.float32) + noise
    if kind == "tv":
        tone = 180.0 * np.sin(2 * np.pi * 440.0 * np.arange(pcm.size) / 8000.0)
        mixed += tone
    return np.clip(np.rint(mixed), -32768, 32767).astype(np.int16)


async def generate(conditions: list[str]) -> list[dict]:
    from app.tts.kokoro import KokoroTTS

    manifest = json.loads((ROOT / "expected_transcripts.json").read_text(encoding="utf-8"))
    tts = KokoroTTS(Settings())
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for item in manifest["items"]:
        pcm8k = await tts.synthesize(item["text"])
        telephone = mulaw_to_pcm16(pcm16_to_mulaw(pcm8k))
        for condition in conditions:
            degraded = _mix_noise(telephone, condition, rng)
            relative = f"{item['id']}__{condition}.wav"
            _write_wav(AUDIO_DIR / relative, degraded, 8000)
            rows.append(
                {
                    "id": f"{item['id']}__{condition}",
                    "source_id": item["id"],
                    "text": item["text"],
                    "tags": item.get("tags") or [],
                    "condition": condition,
                    "path": f"audio/{relative}",
                    "encoding": "pcm16-8k-after-mulaw",
                }
            )
    (AUDIO_DIR / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="clean,street,tv")
    args = parser.parse_args()
    import asyncio

    rows = asyncio.run(generate([item.strip() for item in args.conditions.split(",") if item.strip()]))
    print(f"corpus={len(rows)} dir={AUDIO_DIR}")


if __name__ == "__main__":
    main()
