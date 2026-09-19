from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any


class SessionRegistry:
    def __init__(self) -> None:
        self._ids: set[str] = set()

    def count(self) -> int:
        return len(self._ids)

    def register(self, session_id: str) -> int:
        self._ids.add(session_id)
        return self.count()

    def release(self, session_id: str) -> int:
        self._ids.discard(session_id)
        return self.count()


SESSIONS = SessionRegistry()


def _cpu_percent() -> float | None:
    try:
        load = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        return round((load / cores) * 100.0, 2)
    except (OSError, AttributeError):
        return None


def _ram_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return round(float(parts[1]) / 1024.0, 2)
    except OSError:
        return None
    return None


def _gpu_stats() -> tuple[float | None, float | None]:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return None, None
    try:
        output = subprocess.check_output(
            [
                nvidia,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    first = (output.splitlines() or [""])[0]
    parts = [item.strip() for item in first.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def sample_resources(*, concurrent_sessions: int | None = None) -> dict[str, Any]:
    vram, util = _gpu_stats()
    return {
        "cpu_percent": _cpu_percent(),
        "ram_rss_mb": _ram_rss_mb(),
        "gpu_vram_mb": vram,
        "gpu_utilization": util,
        "concurrent_sessions": SESSIONS.count() if concurrent_sessions is None else concurrent_sessions,
    }
