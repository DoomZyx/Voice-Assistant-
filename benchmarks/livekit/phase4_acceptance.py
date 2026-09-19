from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from livekit_agent.config import get_settings
from livekit_agent.phase4 import phase4_readiness


CHECKLIST = (
    "appel entrant reel sur DID de test",
    "audio entrant fonctionnel",
    "STT francais",
    "reponse vLLM self-hosted",
    "TTS Kokoro streame vers l'appelant",
    "plusieurs tours de conversation",
    "interruption / barge-in",
    "fin de session propre",
    "logs structures call_id/session_id/restaurant_id",
    "latences principales mesurees",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Controle de configuration Phase 4")
    parser.parse_args()
    settings = get_settings()
    report = phase4_readiness(settings)
    print(f"ready={report.ready}")
    if report.missing:
        print("missing=" + ",".join(report.missing))
    if report.blockers:
        print("blockers=" + "; ".join(report.blockers))
    print("checklist_manuelle:")
    for item in CHECKLIST:
        print(f"- [ ] {item}")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
