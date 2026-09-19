from types import SimpleNamespace

from livekit_agent.observability.metrics import CallObserver
from livekit_agent.session import CallIdentity


def test_observer_logs_first_transcript_and_livekit_metrics(caplog):
    identity = CallIdentity(
        call_id="CA1",
        session_id="call-1",
        restaurant_id="tenant-1",
    )
    observer = CallObserver(identity)
    with caplog.at_level("INFO"):
        observer.on_user_transcript("bonjour", is_final=True)
        observer.on_user_transcript("bonjour encore", is_final=True)
        observer.on_livekit_metrics(
            SimpleNamespace(
                ttft=0.12,
                duration=0.4,
                completion_tokens=20,
                prompt_tokens=10,
                tokens_per_second=50.0,
            )
        )
        observer.on_livekit_metrics(SimpleNamespace(ttfb=0.08, duration=0.3, audio_duration=1.2))
        observer.on_session_end()

    joined = "\n".join(caplog.messages)
    assert joined.count("step=time_to_first_transcript") == 1
    assert "call_id=CA1" in joined
    assert "session_id=call-1" in joined
    assert "restaurant_id=tenant-1" in joined
    assert "step=llm.time_to_first_token" in joined
    assert "step=tts.time_to_first_audio" in joined
    assert "step=session.end" in joined
