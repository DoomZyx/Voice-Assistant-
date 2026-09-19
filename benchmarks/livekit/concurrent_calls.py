from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CONCURRENCY_LEVELS = (1, 5, 10, 15, 20, 30, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harness de concurrence LiveKit. Aucune capacite n'est affirmee sans mesure."
    )
    parser.add_argument("--calls", type=int, default=1, help="Nombre d'appels cibles")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/livekit_concurrency.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calls not in CONCURRENCY_LEVELS:
        print(
            f"Niveau hors grille recommandee {CONCURRENCY_LEVELS}. "
            "Le run reste possible, sans claim de capacite."
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_calls": args.calls,
        "grid": list(CONCURRENCY_LEVELS),
        "status": "not_run",
        "note": (
            "Remplacer status apres un bench reel. "
            "Ne pas remplir concurrent_calls_per_GPU avant mesure."
        ),
        "metrics": {
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "gpu_utilization_avg": None,
            "gpu_vram_mb_avg": None,
            "cost_per_concurrent_call": None,
            "cost_per_restaurant": None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
