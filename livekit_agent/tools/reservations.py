from __future__ import annotations

from typing import Any


def not_implemented(name: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": "NOT_IMPLEMENTED",
        "message": f"{name} n'est pas active sur le chemin LiveKit pour le moment.",
    }
