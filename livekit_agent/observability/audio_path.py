from __future__ import annotations

from typing import Any

from livekit_agent.observability.logging import emit_event
from livekit_agent.session import CallIdentity


def _sip_codec(attributes: dict[str, Any]) -> str:
    for key in ("sip.codec", "sip.audioCodec", "sip.twilio.codec", "sip.mediaCodec"):
        value = attributes.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def log_session_audio_contract(
    identity: CallIdentity,
    *,
    sip_attributes: dict[str, Any],
    kokoro_hz: int,
) -> None:
    emit_event(
        "audio.path",
        sip_codec=_sip_codec(sip_attributes),
        pstn_typical_hz=8000,
        inbound_attr_keys=",".join(sorted(str(key) for key in sip_attributes)) or "-",
        trunk_number=identity.trunk_number,
        whisper_target_hz=16000,
        kokoro_native_hz=kokoro_hz,
        livekit_rtc_typical_hz=48000,
        call_id=identity.call_id,
        session_id=identity.session_id,
        restaurant_id=identity.restaurant_id or None,
    )


def log_stt_resample(*, inbound_hz: int, channels: int, whisper_hz: int = 16000) -> None:
    emit_event(
        "audio.stt.resample",
        inbound_hz=inbound_hz,
        channels=channels,
        whisper_hz=whisper_hz,
        resampled=int(inbound_hz) != int(whisper_hz),
    )


def log_tts_resample(*, kokoro_hz: int, emit_hz: int) -> None:
    emit_event(
        "audio.tts.resample",
        kokoro_hz=kokoro_hz,
        emit_hz=emit_hz,
        resampled=int(kokoro_hz) != int(emit_hz),
    )
