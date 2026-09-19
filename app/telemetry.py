"""Metriques vocales en memoire, sans donnees personnelles."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from threading import Lock


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((quantile / 100) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


@dataclass
class _Latency:
    count: int = 0
    total_ms: int = 0
    last_ms: int = 0
    min_ms: int | None = None
    max_ms: int = 0
    samples: deque[int] = field(default_factory=lambda: deque(maxlen=256))

    def add(self, duration_ms: int) -> None:
        value = max(0, int(duration_ms))
        self.count += 1
        self.total_ms += value
        self.last_ms = value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = max(self.max_ms, value)
        self.samples.append(value)

    def snapshot(self) -> dict[str, int]:
        recent = list(self.samples)
        return {
            "count": self.count,
            "average": round(self.total_ms / self.count) if self.count else 0,
            "last": self.last_ms,
            "min": self.min_ms or 0,
            "max": self.max_ms,
            "p50": _percentile(recent, 50),
            "p95": _percentile(recent, 95),
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
