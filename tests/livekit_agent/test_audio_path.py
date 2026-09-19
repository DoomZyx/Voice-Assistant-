from livekit_agent.observability.audio_path import (
    log_session_audio_contract,
    log_stt_resample,
    log_tts_resample,
)
from livekit_agent.session import CallIdentity


def test_audio_path_logs_rates_and_unknown_codec(caplog):
    identity = CallIdentity(call_id="CA1", session_id="call-1", restaurant_id="t1")
    with caplog.at_level("INFO"):
        log_session_audio_contract(
            identity,
            sip_attributes={"sip.trunkPhoneNumber": "+33111"},
            kokoro_hz=24000,
        )
        log_stt_resample(inbound_hz=48000, channels=1, whisper_hz=16000)
        log_tts_resample(kokoro_hz=24000, emit_hz=24000)
    text = " ".join(caplog.messages)
    assert "step=audio.path" in text
    assert "sip_codec=unknown" in text
    assert "whisper_target_hz=16000" in text
    assert "kokoro_native_hz=24000" in text
    assert "inbound_hz=48000" in text
    assert "resampled=True" in text
    assert "resampled=False" in text
