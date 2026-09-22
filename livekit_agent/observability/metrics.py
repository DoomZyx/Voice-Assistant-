from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from livekit_agent.observability.logging import emit_event
from livekit_agent.observability.resources import sample_resources
from livekit_agent.session import CallIdentity


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((quantile / 100) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


class LatencyBook:
    """Histogrammes bornes du worker LiveKit, sans donnees d'appel."""

    def __init__(self, max_samples: int = 256) -> None:
        self._samples: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = Lock()

    def record(self, stage: str, latency_ms: int) -> dict[str, int]:
        with self._lock:
            samples = self._samples[stage]
            samples.append(max(0, int(latency_ms)))
            values = list(samples)
        return {
            "sample_count": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
        }


LATENCIES = LatencyBook()


class CallObserver:
    def __init__(self, identity: CallIdentity) -> None:
        self.identity = identity
        self.connected_at = time.perf_counter()
        self._first_transcript = False

    def _ids(self) -> dict[str, str]:
        return {
            "call_id": self.identity.call_id,
            "session_id": self.identity.session_id,
            "restaurant_id": self.identity.restaurant_id,
        }

    def on_user_transcript(self, text: str, is_final: bool) -> None:
        if not is_final or self._first_transcript:
            return
        self._first_transcript = True
        latency_ms = _ms(time.perf_counter() - self.connected_at)
        emit_event(
            "time_to_first_transcript",
            latency_ms=latency_ms,
            input_characters=len((text or "").strip()),
            **LATENCIES.record("time_to_first_transcript", latency_ms),
            **self._ids(),
        )

    def on_livekit_metrics(self, metrics: Any) -> None:
        ids = self._ids()
        ttft = getattr(metrics, "ttft", None)
        ttfb = getattr(metrics, "ttfb", None)
        duration = getattr(metrics, "duration", None)
        type_name = type(metrics).__name__.lower()
        if ttft is not None:
            tokens_per_sec = getattr(metrics, "tokens_per_second", None)
            completion = getattr(metrics, "completion_tokens", None)
            if tokens_per_sec is None and completion and duration:
                tokens_per_sec = float(completion) / float(duration)
            latency_ms = _ms(float(ttft))
            emit_event(
                "llm.time_to_first_token",
                latency_ms=latency_ms,
                tokens_per_sec=round(float(tokens_per_sec), 2) if tokens_per_sec else None,
                duration_ms=_ms(float(duration)) if duration is not None else None,
                **LATENCIES.record("llm_ttft", latency_ms),
                **ids,
            )
            return
        if ttfb is not None:
            latency_ms = _ms(float(ttfb))
            emit_event(
                "tts.time_to_first_audio",
                latency_ms=latency_ms,
                audio_duration_ms=_ms(float(getattr(metrics, "audio_duration", 0) or 0)),
                **LATENCIES.record("tts_ttfa_excluding_load", latency_ms),
                **ids,
            )
            return
        if duration is not None and ("stt" in type_name or getattr(metrics, "audio_duration", None) is not None):
            latency_ms = _ms(float(duration))
            emit_event(
                "stt.latency",
                latency_ms=latency_ms,
                **LATENCIES.record("stt", latency_ms),
                **ids,
            )

    def on_tool_result(self, name: str, latency_ms: int, success: bool) -> None:
        emit_event(
            "tool_call.latency",
            latency_ms=latency_ms,
            tool=name,
            success=success,
            **self._ids(),
        )

    def on_session_end(self) -> None:
        resources = sample_resources()
        emit_event(
            "session.end",
            latency_ms=_ms(time.perf_counter() - self.connected_at),
            end_to_end_ms=_ms(time.perf_counter() - self.connected_at),
            **self._ids(),
            **resources,
        )
