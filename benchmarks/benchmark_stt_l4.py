"""A/B STT L4. N'invente aucune latence/WER si le GPU est absent."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CANDIDATES = (
    {
        "id": "whisper-small-cuda",
        "role": "baseline historique",
        "model": "small",
        "device": "cuda",
        "streaming": False,
    },
    {
        "id": "whisper-large-v3-turbo",
        "role": "baseline Whisper qualitative",
        "model": "large-v3-turbo",
        "device": "cuda",
        "streaming": False,
    },
    {
        "id": "nemotron-3.5-asr-streaming-0.6b",
        "role": "candidat streaming principal",
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "device": "cuda",
        "streaming": True,
        "chunks_ms": [80, 160, 320, 560, 1120],
    },
    {
        "id": "parakeet-tdt-0.6b-v3",
        "role": "candidat qualite/perf supplementaire",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "device": "cuda",
        "streaming": False,
    },
)

REQUIRED_METRICS = (
    "partial_first_ms",
    "partial_stable_ms",
    "final_ms",
    "wer_final",
    "wer_partial",
    "rtf",
    "gpu_memory_mib",
    "gpu_utilization",
)


def probe_gpu() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi absent"}
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True, "devices": listing.stdout.strip()}


def main() -> None:
    gpu = probe_gpu()
    payload = {
        "hardware": gpu,
        "order": [item["id"] for item in CANDIDATES],
        "required_metrics": list(REQUIRED_METRICS),
        "note": (
            "Ne pas juger les orchestrateurs sur speech_end->first_audio tant que "
            "le STT n'est pas en temps reel. Mesurer partiels puis WER final."
        ),
        "candidates": [],
    }
    for item in CANDIDATES:
        row = dict(item)
        if not gpu["available"]:
            row["status"] = "not measured: no NVIDIA GPU on this machine"
            for metric in REQUIRED_METRICS:
                row[metric] = None
        else:
            row["status"] = "not measured: runner not executed yet"
            for metric in REQUIRED_METRICS:
                row[metric] = None
        payload["candidates"].append(row)
    out = ROOT / "results" / "stt_l4_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
