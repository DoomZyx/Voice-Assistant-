"""Sweep des seuils de barge-in. N'invente aucun chiffre GPU/Twilio live."""

from __future__ import annotations

import json
from pathlib import Path

from app.session.interrupt import sweep_confirm_thresholds


ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = sweep_confirm_thresholds()
    chosen = next(row for row in rows if row["confirm_ms"] == 160)
    payload = {
        "hardware": "control-plane",
        "default_confirm_ms": 160,
        "default_is_optimum": False,
        "reason": (
            "160 est dans la bande 120-200 demandee. "
            "Il coupe le burst 50 ms et garde une parole continue 200 ms. "
            "Le bruit soutenu 250 ms reste une interruption si le VAD le classe parole."
        ),
        "rows": rows,
        "chosen": chosen,
    }
    out = ROOT / "results" / "barge_in_confirm.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
