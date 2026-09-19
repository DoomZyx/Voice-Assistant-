from __future__ import annotations

import time
from typing import Any

from livekit_agent.observability.logging import emit_event
from livekit_agent.observability.resources import sample_resources
from livekit_agent.session import CallIdentity


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


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
        emit_event(
            "time_to_first_transcript",
            latency_ms=_ms(time.perf_counter() - self.connected_at),
            text=(text or "").strip()[:80] or None,
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
            emit_event(
                "llm.time_to_first_token",
                latency_ms=_ms(float(ttft)),
                tokens_per_sec=round(float(tokens_per_sec), 2) if tokens_per_sec else None,
                duration_ms=_ms(float(duration)) if duration is not None else None,
                **ids,
            )
            return
        if ttfb is not None:
            emit_event(
                "tts.time_to_first_audio",
                latency_ms=_ms(float(ttfb)),
                audio_duration_ms=_ms(float(getattr(metrics, "audio_duration", 0) or 0)),
                **ids,
            )
            return
        if duration is not None and ("stt" in type_name or getattr(metrics, "audio_duration", None) is not None):
            emit_event(
                "stt.latency",
                latency_ms=_ms(float(duration)),
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
