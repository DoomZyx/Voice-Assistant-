"""Metriques vocales en memoire, sans donnees personnelles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import Lock


@dataclass
class _Latency:
    count: int = 0
    total_ms: int = 0
    last_ms: int = 0
    min_ms: int | None = None
    max_ms: int = 0

    def add(self, duration_ms: int) -> None:
        value = max(0, int(duration_ms))
        self.count += 1
        self.total_ms += value
        self.last_ms = value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = max(self.max_ms, value)

    def snapshot(self) -> dict[str, int]:
        return {
            "count": self.count,
            "average": round(self.total_ms / self.count) if self.count else 0,
            "last": self.last_ms,
            "min": self.min_ms or 0,
            "max": self.max_ms,
        }


class VoiceTelemetry:
    """Registre borne aux compteurs et latences agreges du processus."""

    def __init__(self) -> None:
        self._latencies: dict[str, _Latency] = {}
        self._errors: Counter[str] = Counter()
        self._events: Counter[str] = Counter()
        self._lock = Lock()

    def record_latency(self, stage: str, duration_ms: int) -> None:
        with self._lock:
            self._latencies.setdefault(stage, _Latency()).add(duration_ms)

    def record_error(self, stage: str) -> None:
        with self._lock:
            self._errors[stage] += 1

    def record_event(self, event: str) -> None:
        with self._lock:
            self._events[event] += 1

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                "latency_ms": {
                    stage: latency.snapshot()
                    for stage, latency in sorted(self._latencies.items())
                },
                "errors": dict(sorted(self._errors.items())),
                "events": dict(sorted(self._events.items())),
            }
